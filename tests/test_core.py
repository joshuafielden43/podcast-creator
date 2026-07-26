"""
Tests for core utility functions
"""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from podcast_creator.core import (
    canonicalize_speaker_labels,
    create_outline_parser,
    create_validated_transcript_schema,
    outline_parser,
    clean_thinking_content,
    create_validated_transcript_parser,
    extract_text_content,
    interleave_segment_gaps,
    make_silence_clip,
    parse_thinking_content,
    has_long_silence,
    is_usable_audio,
    trim_trailing_silence,
)


class TestSegmentBoundaryGaps:
    def test_inserts_silence_after_segment_ends_but_not_after_final_clip(self):
        class FakeClip:
            def __init__(self, name):
                self.name = name
                self.fps = 44100
                self.nchannels = 1

        clips = [FakeClip("a"), FakeClip("b"), FakeClip("c"), FakeClip("d")]
        # Two outline segments: clips 0-1 and 2-3
        timeline = interleave_segment_gaps(
            clips, segment_end_indices=[1, 3], gap_seconds=1.0
        )

        assert len(timeline) == 5  # four speech + one gap after first segment
        assert timeline[0] is clips[0]
        assert timeline[1] is clips[1]
        assert timeline[2] is not clips[2]  # silence
        assert getattr(timeline[2], "duration", None) == 1.0
        assert timeline[3] is clips[2]
        assert timeline[4] is clips[3]

    def test_no_gaps_without_boundaries(self):
        clips = [object(), object()]
        assert interleave_segment_gaps(clips, segment_end_indices=None) == clips
        assert interleave_segment_gaps(clips, segment_end_indices=[]) == clips

    def test_make_silence_clip_duration(self):
        silence = make_silence_clip(1.0, fps=44100, nchannels=1)
        try:
            assert abs(silence.duration - 1.0) < 1e-6
        finally:
            silence.close()


class TestTrailingSilenceTrim:
    def test_replaces_a_clip_with_its_trimmed_version(self, tmp_path, monkeypatch):
        clip = tmp_path / "clip.mp3"
        clip.write_bytes(b"padded")

        def fake_run(command, **_):
            if command[-1] != "-":
                Path(command[-1]).write_bytes(b"trimmed")

        monkeypatch.setattr("podcast_creator.core.subprocess.run", fake_run)
        monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")

        trim_trailing_silence(clip)

        assert clip.read_bytes() == b"trimmed"


class TestAudioValidation:
    def test_flags_missing_or_silent_clips(self, tmp_path, monkeypatch):
        assert has_long_silence(tmp_path / "missing.mp3")

        clip = tmp_path / "clip.mp3"
        clip.write_bytes(b"audio")
        monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
        monkeypatch.setattr(
            "podcast_creator.core.subprocess.run",
            lambda *_args, **_kwargs: CompletedProcess([], 0, stderr="silence_start: 0"),
        )

        assert has_long_silence(clip)

    def test_rejects_implausibly_short_audio(self, tmp_path, monkeypatch):
        clip = tmp_path / "clip.mp3"
        clip.write_bytes(b"audio")
        monkeypatch.setattr("podcast_creator.core.has_long_silence", lambda _: False)

        class FakeClip:
            duration = 0.48

            def close(self):
                pass

        monkeypatch.setattr("podcast_creator.core.AudioFileClip", lambda _: FakeClip())

        assert not is_usable_audio(clip, "one two three four five six seven")

    def test_rejects_truncated_clips_that_pass_the_old_0_08_floor(self, tmp_path, monkeypatch):
        # 7 words * 0.08 = 0.56s would have passed; 0.30 s/word needs 2.1s.
        clip = tmp_path / "clip.mp3"
        clip.write_bytes(b"audio")
        monkeypatch.setattr("podcast_creator.core.has_long_silence", lambda _: False)

        class FakeClip:
            duration = 1.0

            def close(self):
                pass

        monkeypatch.setattr("podcast_creator.core.AudioFileClip", lambda _: FakeClip())

        assert not is_usable_audio(clip, "one two three four five six seven")


class TestOutlineParser:
    def test_accepts_outline_wrapper(self):
        outline = outline_parser.parse(
            '{"outline": {"segments": [{"name": "One", "description": "First", "size": "short"}]}}'
        )

        assert outline.segments[0].name == "One"

    def test_requires_the_selected_segment_count_and_size(self):
        parser = create_outline_parser(2)

        assert (
            len(
                parser.parse("""{"segments": [
            {"name": "One", "description": "First", "size": "short"},
            {"name": "Two", "description": "Second", "size": "long"}
        ]}""").segments
            )
            == 2
        )

        import pytest

        with pytest.raises(Exception):
            parser.parse(
                '{"segments": [{"name": "One", "description": "First", "size": "short"}]}'
            )

        with pytest.raises(Exception):
            parser.parse(
                """{"segments": [
                    {"name": "One", "description": "First"},
                    {"name": "Two", "description": "Second", "size": "long"}
                ]}"""
            )


class TestExtractTextContent:
    """Tests for extract_text_content function"""

    def test_string_passthrough(self):
        """Plain string content is returned as-is"""
        assert extract_text_content("hello world") == "hello world"

    def test_gemini_format(self):
        """Structured list with dict parts containing 'text' key (Gemini-style)"""
        content = [{"type": "text", "text": "hello world", "extras": {"some": "data"}}]
        assert extract_text_content(content) == "hello world"

    def test_gemini_format_multiple_parts(self):
        """Multiple structured dict parts are concatenated"""
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert extract_text_content(content) == "hello world"

    def test_list_of_strings(self):
        """List of plain strings is concatenated"""
        assert extract_text_content(["hello", " world"]) == "hello world"

    def test_mixed_list(self):
        """Mixed list of dicts and strings is handled correctly"""
        content = [
            {"type": "text", "text": "hello "},
            "world",
        ]
        assert extract_text_content(content) == "hello world"

    def test_empty_string(self):
        """Empty string returns empty string"""
        assert extract_text_content("") == ""

    def test_empty_list(self):
        """Empty list returns empty string"""
        assert extract_text_content([]) == ""

    def test_none(self):
        """None returns empty string"""
        assert extract_text_content(None) == ""

    def test_non_string_non_list_fallback(self):
        """Non-string, non-list types fall back to str()"""
        assert extract_text_content(42) == "42"

    def test_dict_without_text_key_skipped(self):
        """Dict items without 'text' key are skipped"""
        content = [
            {"type": "image", "url": "http://example.com"},
            {"type": "text", "text": "hello"},
        ]
        assert extract_text_content(content) == "hello"


class TestParseThinkingContent:
    """Tests for parse_thinking_content and clean_thinking_content"""

    def test_closed_think_tags(self):
        """Standard <think>...</think> tags are removed"""
        content = '<think>Let me analyze this</think>{"answer": "yes"}'
        thinking, cleaned = parse_thinking_content(content)
        assert thinking == "Let me analyze this"
        assert cleaned == '{"answer": "yes"}'

    def test_no_think_tags(self):
        """Content without think tags is returned as-is"""
        content = '{"transcript": [{"speaker": "Alice", "dialogue": "Hello"}]}'
        thinking, cleaned = parse_thinking_content(content)
        assert thinking == ""
        assert cleaned == content

    def test_unclosed_think_tag_json_same_line(self):
        """Unclosed <think> with JSON starting on same line as thinking text"""
        content = (
            "<think>The user wants a podcast transcript.\n"
            "I need to focus on the key points.\n"
            "Speaker Roles:\n"
            "- Dr. Alex: Analytical\n"
            "- Jamie: Enthusiastic\n"
            "\n"
            "This will ensure at least 3 turns and cover all points."
            '{"transcript": [{"speaker": "Alice", "dialogue": "Hello"}]}'
        )
        thinking, cleaned = parse_thinking_content(content)
        assert "<think>" not in cleaned
        assert '"transcript"' in cleaned
        assert "The user wants" in thinking

    def test_unclosed_think_tag_json_new_line(self):
        """Unclosed <think> without </think> followed by JSON on new line"""
        content = (
            "<think>\n"
            "Let me think about this carefully.\n"
            "I should create a natural dialogue.\n"
            "\n"
            '{"transcript": [{"speaker": "Alice", "dialogue": "Hi there!"}]}'
        )
        thinking, cleaned = parse_thinking_content(content)
        assert "<think>" not in cleaned
        assert '"transcript"' in cleaned
        assert "think about this carefully" in thinking

    def test_closed_think_tag_multiline(self):
        """Closed <think> tag with multi-line thinking then JSON"""
        content = (
            "<think>\n"
            "The user wants a podcast transcript for the first segment.\n"
            "I need to focus on:\n"
            "1. Introducing the topic\n"
            "2. Discussing the key points\n"
            "</think>\n"
            '{"transcript": [{"speaker": "Dr. Alex", "dialogue": "Welcome!"}]}'
        )
        thinking, cleaned = parse_thinking_content(content)
        assert "<think>" not in cleaned
        assert '"transcript"' in cleaned

    def test_unclosed_think_tag_no_json(self):
        """Unclosed <think> with no JSON content after it"""
        content = "<think>Just thinking, no output"
        thinking, cleaned = parse_thinking_content(content)
        assert "<think>" not in cleaned
        assert "Just thinking" in thinking

    def test_clean_thinking_content_unclosed(self):
        """clean_thinking_content convenience function handles unclosed tags"""
        content = (
            "<think>\n"
            "Some thinking here.\n"
            "\n"
            '{"transcript": [{"speaker": "Alice", "dialogue": "Hello"}]}'
        )
        cleaned = clean_thinking_content(content)
        assert "<think>" not in cleaned
        assert '"transcript"' in cleaned

    def test_non_string_input(self):
        """Non-string input is handled gracefully"""
        thinking, cleaned = parse_thinking_content(None)
        assert thinking == ""
        assert cleaned == ""

    def test_multiple_closed_think_tags(self):
        """Multiple closed think blocks are all removed"""
        content = "<think>First</think>Hello <think>Second</think>World"
        thinking, cleaned = parse_thinking_content(content)
        assert "First" in thinking
        assert "Second" in thinking
        assert cleaned == "Hello World"

    def test_realistic_deepseek_response(self):
        """Realistic DeepSeek-style response with long thinking and no closing tag"""
        content = (
            "<think>The user wants a podcast transcript for the first segment "
            "of an episode about SurrealDB 3.0.\n"
            "I need to focus on:\n"
            "1.  **Introducing SurrealDB 3.0's vision.**\n"
            "2.  **Discussing the pain points of multi-model applications.**\n"
            "\n"
            "**Speaker Roles:**\n"
            "*   **Dr. Alex Chen:** Senior AI researcher\n"
            "*   **Jamie Rodriguez:** Full-stack engineer\n"
            "\n"
            "This will ensure at least 3 turns and cover all points."
            '{"transcript": [\n'
            "    {\n"
            '        "speaker": "Dr. Alex Chen",\n'
            '        "dialogue": "Welcome to the podcast, everyone."\n'
            "    },\n"
            "    {\n"
            '        "speaker": "Jamie Rodriguez",\n'
            '        "dialogue": "Thanks for having me!"\n'
            "    }\n"
            "]}"
        )
        thinking, cleaned = parse_thinking_content(content)
        assert "<think>" not in cleaned
        assert '"transcript"' in cleaned
        assert "SurrealDB 3.0" in thinking
        # Verify the JSON is parseable
        import json

        parsed = json.loads(cleaned)
        assert len(parsed["transcript"]) == 2


class TestCanonicalizeSpeakerLabels:
    """Structural speaker recovery: aliases, solo residual, equal-count casts."""

    def test_exact_names_passthrough(self):
        mapping = canonicalize_speaker_labels(
            ["Dr. Alex Chen", "Jamie Rodriguez"],
            ["Dr. Alex Chen", "Jamie Rodriguez"],
        )
        assert mapping == {
            "Dr. Alex Chen": "Dr. Alex Chen",
            "Jamie Rodriguez": "Jamie Rodriguez",
        }

    def test_multi_token_alias_of_full_name(self):
        # Production failure shape: LLM drops the middle name on one row.
        mapping = canonicalize_speaker_labels(
            ["Professor Sarah Kim", "Professor Kim"],
            ["Professor Sarah Kim"],
        )
        assert mapping == {
            "Professor Sarah Kim": "Professor Sarah Kim",
            "Professor Kim": "Professor Sarah Kim",
        }

    def test_solo_profile_maps_any_label(self):
        mapping = canonicalize_speaker_labels(
            ["The Host", "Narrator", "Professor Kim"],
            ["Professor Sarah Kim"],
        )
        assert set(mapping.values()) == {"Professor Sarah Kim"}
        assert len(mapping) == 3

    def test_single_token_nickname(self):
        mapping = canonicalize_speaker_labels(
            ["Alex", "Jamie"],
            ["Dr. Alex Chen", "Jamie Rodriguez"],
        )
        assert mapping == {
            "Alex": "Dr. Alex Chen",
            "Jamie": "Jamie Rodriguez",
        }

    def test_case_insensitive_exact(self):
        mapping = canonicalize_speaker_labels(
            ["professor sarah kim"],
            ["Professor Sarah Kim"],
        )
        assert mapping == {"professor sarah kim": "Professor Sarah Kim"}

    def test_equal_count_cast_replacement_preserves_work(self):
        mapping = canonicalize_speaker_labels(
            ["Host", "Guest"],
            ["Dr. Alex Chen", "Jamie Rodriguez"],
        )
        assert mapping == {
            "Host": "Dr. Alex Chen",
            "Guest": "Jamie Rodriguez",
        }

    def test_partial_match_plus_equal_count_residual(self):
        mapping = canonicalize_speaker_labels(
            ["Alex", "Jordan"],
            ["Dr. Alex Chen", "Jamie Rodriguez"],
        )
        assert mapping == {
            "Alex": "Dr. Alex Chen",
            "Jordan": "Jamie Rodriguez",
        }

    def test_ambiguous_shared_token_rejects(self):
        with pytest.raises(ValueError, match="Invalid speaker names: Kim"):
            canonicalize_speaker_labels(
                ["Kim"],
                ["Sarah Kim", "John Kim"],
            )

    def test_wrong_cardinality_rejects(self):
        with pytest.raises(ValueError, match="Invalid speaker names: Sam"):
            canonicalize_speaker_labels(
                ["Sam"],
                ["Dr. Alex Chen", "Jamie Rodriguez"],
            )

    def test_title_period_normalization(self):
        mapping = canonicalize_speaker_labels(
            ["Dr Alex"],
            ["Dr. Alex Chen", "Jamie Rodriguez"],
        )
        assert mapping == {"Dr Alex": "Dr. Alex Chen"}


class TestValidatedTranscriptParser:
    def test_schema_requires_configured_speakers(self):
        schema = create_validated_transcript_schema(["Dr. Alex Chen"])

        assert schema.model_validate({
            "transcript": [{"speaker": "Dr. Alex Chen", "dialogue": "Hello."}]
        })

        with pytest.raises(Exception):
            schema.model_validate({
                "transcript": [{"speaker": "Alex", "dialogue": "Hello."}]
            })

    def test_canonicalizes_a_complete_replacement_cast(self):
        parser = create_validated_transcript_parser([
            "Dr. Alex Chen",
            "Jamie Rodriguez",
        ])

        transcript = parser.parse(
            '{"transcript": ['
            '{"speaker": "Alex", "dialogue": "Welcome."}, '
            '{"speaker": "Jordan", "dialogue": "Thanks."}'
            "]}"
        )

        assert [dialogue.speaker for dialogue in transcript.transcript] == [
            "Dr. Alex Chen",
            "Jamie Rodriguez",
        ]

    def test_canonicalizes_mixed_exact_and_multi_token_alias(self):
        parser = create_validated_transcript_parser(["Professor Sarah Kim"])

        transcript = parser.parse(
            '{"transcript": ['
            '{"speaker": "Professor Sarah Kim", "dialogue": "Welcome."}, '
            '{"speaker": "Professor Kim", "dialogue": "Continuing."}, '
            '{"speaker": "Professor Sarah Kim", "dialogue": "Wrap up."}'
            "]}"
        )

        assert [dialogue.speaker for dialogue in transcript.transcript] == [
            "Professor Sarah Kim",
            "Professor Sarah Kim",
            "Professor Sarah Kim",
        ]

    def test_rejects_an_ambiguous_replacement_speaker(self):
        parser = create_validated_transcript_parser([
            "Dr. Alex Chen",
            "Jamie Rodriguez",
        ])

        with pytest.raises(Exception, match="Invalid speaker names: Sam"):
            parser.parse('{"transcript": [{"speaker": "Sam", "dialogue": "Hello."}]}')
