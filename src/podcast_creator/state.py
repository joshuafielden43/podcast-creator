from operator import add
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict, Union

from .core import Dialogue, Outline
from .speakers import SpeakerProfile


class PodcastState(TypedDict):
    # Input data
    content: Union[str, List[str]]
    briefing: str
    num_segments: int
    language: Optional[str]

    # Generated content
    outline: Optional[Outline]
    transcript: List[Dialogue]
    # 0-based index of the last dialogue clip in each outline segment
    segment_end_indices: List[int]

    # Audio processing
    audio_clips: Annotated[List[Path], add]
    final_output_file_path: Optional[Path]

    # Configuration
    output_dir: Path
    episode_name: str
    speaker_profile: Optional[SpeakerProfile]
