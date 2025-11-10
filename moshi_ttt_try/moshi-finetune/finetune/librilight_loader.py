"""
LibriLight Data Loader for TTT Long-Context Evaluation

This module provides functionality to load and concatenate LibriLight audiobook chapters
for testing TTT's long-context capabilities following the TTT paper methodology.

Key Features:
- Discover books and chapters from LibriLight dataset
- Concatenate consecutive chapters for long sequences
- Rotate through different books/speakers for robust evaluation
- Rich metadata support for book selection and filtering
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class LibriLightBook:
    """Represents a single audiobook with its chapters and metadata."""
    
    def __init__(self, book_path: Path):
        self.book_path = book_path
        self.book_name = book_path.name
        self.speaker_id = book_path.parent.name
        self.chapters = []
        self.metadata = {}
        self._load_chapters()
        self._load_metadata()
    
    def _load_chapters(self):
        """Load all chapter files (.flac) from the book directory."""
        chapter_files = sorted(list(self.book_path.glob("*.flac")))
        for chapter_file in chapter_files:
            chapter_info = {
                'path': chapter_file,
                'name': chapter_file.stem,
                'size_mb': chapter_file.stat().st_size / (1024 * 1024),
                'json_path': chapter_file.with_suffix('.json')
            }
            self.chapters.append(chapter_info)
        
        logger.debug(f"Loaded {len(self.chapters)} chapters for book {self.book_name}")
    
    def _load_metadata(self):
        """Load metadata from the first chapter's JSON file."""
        if self.chapters and self.chapters[0]['json_path'].exists():
            try:
                with open(self.chapters[0]['json_path'], 'r') as f:
                    self.metadata = json.load(f)
                
                # Extract useful info
                book_meta = self.metadata.get('book_meta', {})
                self.title = book_meta.get('title', 'Unknown')
                self.author = book_meta.get('authors', [{}])[0].get('last_name', 'Unknown')
                self.genre = book_meta.get('genre', ['Unknown'])[0]
                self.total_time_secs = book_meta.get('totaltimesecs', 0)
                self.num_sections = int(book_meta.get('num_sections', len(self.chapters)))
                self.snr = self.metadata.get('snr', 0)
                
                logger.debug(f"Loaded metadata for '{self.title}' by {self.author}")
                
            except Exception as e:
                logger.warning(f"Failed to load metadata for {self.book_name}: {e}")
                self._set_default_metadata()
        else:
            self._set_default_metadata()
    
    def _set_default_metadata(self):
        """Set default metadata when JSON loading fails."""
        self.title = self.book_name
        self.author = "Unknown"
        self.genre = "Unknown"
        self.total_time_secs = 0
        self.num_sections = len(self.chapters)
        self.snr = 0
    
    def get_chapters(self, start_idx: int = 0, num_chapters: Optional[int] = None) -> List[Dict]:
        """Get a subset of chapters starting from start_idx."""
        if num_chapters is None:
            return self.chapters[start_idx:]
        else:
            return self.chapters[start_idx:start_idx + num_chapters]
    
    def get_total_duration_estimate(self, num_chapters: Optional[int] = None) -> float:
        """Estimate total duration in minutes based on file sizes."""
        chapters = self.get_chapters(0, num_chapters)
        total_size_mb = sum(ch['size_mb'] for ch in chapters)
        # Rough estimate: 1MB ≈ 1.5 minutes for 64kbps audio
        return total_size_mb * 1.5
    
    def is_suitable_for_ttt(self) -> bool:
        """Check if this book is suitable for TTT evaluation."""
        # Quality filters for TTT evaluation
        return (
            len(self.chapters) >= 3 and  # At least 3 chapters
            self.snr > 5.0 and  # Reasonable audio quality
            self.total_time_secs > 3600 and  # At least 1 hour total
            'drama' not in self.genre.lower()  # Avoid dramatic readings (multiple speakers)
        )
    
    def __str__(self):
        return f"LibriLightBook(speaker={self.speaker_id}, title='{self.title}', chapters={len(self.chapters)})"


class LibriLightLoader:
    """Main loader class for LibriLight dataset management."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.books = {}
        self.speakers = {}
        self._discover_books()
    
    def _discover_books(self):
        """Discover all books and speakers in the dataset."""
        logger.info(f"Discovering LibriLight books in {self.dataset_path}")
        
        book_count = 0
        speaker_count = 0
        
        # Walk through speaker directories
        for speaker_dir in self.dataset_path.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            speaker_books = []
            
            # Walk through book directories for this speaker
            for book_dir in speaker_dir.iterdir():
                if not book_dir.is_dir():
                    continue
                
                # Check if this directory contains audio files
                if list(book_dir.glob("*.flac")):
                    book = LibriLightBook(book_dir)
                    book_key = f"{speaker_id}/{book.book_name}"
                    self.books[book_key] = book
                    speaker_books.append(book)
                    book_count += 1
            
            if speaker_books:
                self.speakers[speaker_id] = speaker_books
                speaker_count += 1
        
        logger.info(f"Discovered {book_count} books from {speaker_count} speakers")
    
    def get_suitable_books_for_ttt(self) -> List[LibriLightBook]:
        """Get books that are suitable for TTT evaluation."""
        suitable_books = []
        for book in self.books.values():
            if book.is_suitable_for_ttt():
                suitable_books.append(book)
        
        logger.info(f"Found {len(suitable_books)} books suitable for TTT evaluation")
        return suitable_books
    
    def get_book(self, speaker_id: str, book_name: str) -> Optional[LibriLightBook]:
        """Get a specific book by speaker and book name."""
        book_key = f"{speaker_id}/{book_name}"
        return self.books.get(book_key)
    
    def get_speaker_books(self, speaker_id: str) -> List[LibriLightBook]:
        """Get all books for a specific speaker."""
        return self.speakers.get(speaker_id, [])
    
    def get_random_book_sequence(self, num_chapters: int = 3, 
                                min_snr: float = 5.0) -> Optional[Tuple[LibriLightBook, List[Dict]]]:
        """Get a random book sequence suitable for TTT evaluation."""
        suitable_books = [book for book in self.books.values() 
                         if book.is_suitable_for_ttt() and book.snr >= min_snr]
        
        if not suitable_books:
            logger.warning("No suitable books found for TTT evaluation")
            return None
        
        # Select random book and chapters
        book = random.choice(suitable_books)
        max_start = max(0, len(book.chapters) - num_chapters)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        chapters = book.get_chapters(start_idx, num_chapters)
        
        logger.info(f"Selected book '{book.title}' by {book.author}, "
                   f"chapters {start_idx+1}-{start_idx+len(chapters)}")
        
        return book, chapters
    
    def concatenate_audio_files(self, chapter_paths: List[Path], 
                               output_path: Optional[Path] = None) -> Path:
        """
        Concatenate audio files using Python audio libraries (fallback to ffmpeg if available).
        
        Args:
            chapter_paths: List of paths to audio files to concatenate
            output_path: Output path for concatenated file (if None, creates temp file)
        
        Returns:
            Path to the concatenated audio file
        """
        if output_path is None:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='librilight_concat_')
            os.close(temp_fd)  # Close the file descriptor, we just need the path
            output_path = Path(temp_path)
        
        try:
            # Try using pydub first (Python-based, more reliable)
            return self._concatenate_with_pydub(chapter_paths, output_path)
        except Exception as e:
            logger.warning(f"Pydub concatenation failed: {e}, trying ffmpeg...")
            try:
                return self._concatenate_with_ffmpeg(chapter_paths, output_path)
            except Exception as e2:
                logger.error(f"Both pydub and ffmpeg failed. Pydub: {e}, ffmpeg: {e2}")
                raise RuntimeError(f"Audio concatenation failed with both methods")
    
    def _concatenate_with_pydub(self, chapter_paths: List[Path], output_path: Path) -> Path:
        """Concatenate using pydub (Python-based audio processing)."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError("pydub not available for audio concatenation")
        
        logger.debug(f"Concatenating {len(chapter_paths)} files with pydub")
        
        # Load and concatenate audio segments
        combined = None
        for i, chapter_path in enumerate(chapter_paths):
            logger.debug(f"Loading chapter {i+1}/{len(chapter_paths)}: {chapter_path.name}")
            
            # Load audio file (pydub auto-detects format)
            segment = AudioSegment.from_file(str(chapter_path))
            
            if combined is None:
                combined = segment
            else:
                combined += segment
        
        if combined is None:
            raise RuntimeError("No audio segments loaded")
        
        # Export as WAV
        combined.export(str(output_path), format="wav")
        
        logger.info(f"Successfully concatenated {len(chapter_paths)} chapters to {output_path} using pydub")
        return output_path
    
    def _concatenate_with_ffmpeg(self, chapter_paths: List[Path], output_path: Path) -> Path:
        """Fallback concatenation using ffmpeg."""
        # Create ffmpeg input list
        input_list_fd, input_list_path = tempfile.mkstemp(suffix='.txt', prefix='ffmpeg_inputs_')
        try:
            with os.fdopen(input_list_fd, 'w') as f:
                for chapter_path in chapter_paths:
                    f.write(f"file '{chapter_path.absolute()}'\n")
            
            # Run ffmpeg concatenation
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-f', 'concat',
                '-safe', '0',
                '-i', input_list_path,
                '-c', 'copy',  # Copy without re-encoding when possible
                str(output_path)
            ]
            
            logger.debug(f"Running ffmpeg concatenation: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                raise RuntimeError(f"Audio concatenation failed: {result.stderr}")
            
            logger.info(f"Successfully concatenated {len(chapter_paths)} chapters to {output_path} using ffmpeg")
            return output_path
            
        finally:
            # Clean up input list file
            if os.path.exists(input_list_path):
                os.unlink(input_list_path)
    
    def get_ttt_evaluation_chapters(self, 
                                   speaker_id: Optional[str] = None,
                                   book_name: Optional[str] = None,
                                   num_chapters: int = 3) -> Optional[Tuple[List[Path], Dict]]:
        """
        Get chapter files for TTT evaluation (without concatenation).
        
        This approach avoids the need for ffmpeg/audio concatenation by returning
        individual chapter files that can be processed sequentially during evaluation.
        
        Args:
            speaker_id: Specific speaker ID (if None, random selection)
            book_name: Specific book name (if None, random selection)
            num_chapters: Number of chapters to use
        
        Returns:
            Tuple of (chapter_paths_list, metadata_dict) or None if failed
        """
        try:
            # Get book and chapters
            if speaker_id and book_name:
                book = self.get_book(speaker_id, book_name)
                if not book:
                    logger.error(f"Book {book_name} not found for speaker {speaker_id}")
                    return None
                chapters = book.get_chapters(0, num_chapters)
            else:
                result = self.get_random_book_sequence(num_chapters)
                if not result:
                    return None
                book, chapters = result
            
            if len(chapters) < num_chapters:
                logger.warning(f"Only {len(chapters)} chapters available, requested {num_chapters}")
            
            # Get chapter paths
            chapter_paths = [ch['path'] for ch in chapters]
            
            # Create metadata
            metadata = {
                'speaker_id': book.speaker_id,
                'book_name': book.book_name,
                'book_title': book.title,
                'author': book.author,
                'genre': book.genre,
                'num_chapters': len(chapters),
                'chapter_names': [ch['name'] for ch in chapters],
                'estimated_duration_minutes': book.get_total_duration_estimate(len(chapters)),
                'snr': book.snr,
                'source_files': [str(ch['path']) for ch in chapters]
            }
            
            logger.info(f"Selected TTT evaluation chapters: {len(chapter_paths)} chapters")
            logger.info(f"Book: '{book.title}' by {book.author}")
            
            return chapter_paths, metadata
            
        except Exception as e:
            logger.error(f"Failed to get TTT evaluation chapters: {e}")
            return None
    
    def create_ttt_evaluation_sequence(self, 
                                     speaker_id: Optional[str] = None,
                                     book_name: Optional[str] = None,
                                     num_chapters: int = 3,
                                     output_dir: Optional[Path] = None) -> Optional[Tuple[Path, Dict]]:
        """
        Create a concatenated audio sequence for TTT evaluation.
        
        This is a fallback method that requires ffmpeg. Use get_ttt_evaluation_chapters() 
        for a more robust approach that doesn't require audio concatenation.
        
        Args:
            speaker_id: Specific speaker ID (if None, random selection)
            book_name: Specific book name (if None, random selection)
            num_chapters: Number of chapters to concatenate
            output_dir: Directory to save the concatenated file
        
        Returns:
            Tuple of (concatenated_audio_path, metadata_dict) or None if failed
        """
        logger.warning("create_ttt_evaluation_sequence() requires ffmpeg. Consider using get_ttt_evaluation_chapters() instead.")
        
        try:
            # Get chapters first
            result = self.get_ttt_evaluation_chapters(speaker_id, book_name, num_chapters)
            if not result:
                return None
            
            chapter_paths, metadata = result
            
            # Create output path
            if output_dir is None:
                output_dir = Path(tempfile.gettempdir())
            
            output_filename = f"librilight_{metadata['speaker_id']}_{metadata['book_name'][:20]}_{len(chapter_paths)}ch.wav"
            output_path = output_dir / output_filename
            
            # Try to concatenate audio files
            concat_path = self.concatenate_audio_files(chapter_paths, output_path)
            
            logger.info(f"Created TTT evaluation sequence: {concat_path}")
            
            return concat_path, metadata
            
        except Exception as e:
            logger.error(f"Failed to create TTT evaluation sequence: {e}")
            return None
    
    def get_evaluation_rotation(self, num_sequences: int = 5, 
                              num_chapters: int = 3) -> List[Tuple[Path, Dict]]:
        """
        Create multiple evaluation sequences for robust TTT testing (TTT paper style).
        
        Args:
            num_sequences: Number of different sequences to create
            num_chapters: Chapters per sequence
        
        Returns:
            List of (audio_path, metadata) tuples
        """
        sequences = []
        suitable_books = self.get_suitable_books_for_ttt()
        
        if len(suitable_books) < num_sequences:
            logger.warning(f"Only {len(suitable_books)} suitable books found, "
                          f"requested {num_sequences} sequences")
            num_sequences = len(suitable_books)
        
        # Select diverse books (different speakers/genres when possible)
        selected_books = random.sample(suitable_books, num_sequences)
        
        for book in selected_books:
            try:
                chapters = book.get_chapters(0, num_chapters)
                
                # Create temporary concatenated file
                chapter_paths = [ch['path'] for ch in chapters]
                concat_path = self.concatenate_audio_files(chapter_paths)
                
                metadata = {
                    'speaker_id': book.speaker_id,
                    'book_title': book.title,
                    'author': book.author,
                    'genre': book.genre,
                    'num_chapters': len(chapters),
                    'estimated_duration_minutes': book.get_total_duration_estimate(len(chapters))
                }
                
                sequences.append((concat_path, metadata))
                logger.info(f"Added sequence: '{book.title}' by {book.author}")
                
            except Exception as e:
                logger.error(f"Failed to create sequence for book '{book.title}': {e}")
                continue
        
        logger.info(f"Created {len(sequences)} evaluation sequences for rotation")
        return sequences
    
    def cleanup_temp_files(self, file_paths: List[Path]):
        """Clean up temporary files created during evaluation."""
        for file_path in file_paths:
            try:
                if file_path.exists() and '/tmp/' in str(file_path):
                    file_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")


# Example usage and testing functions
def test_librilight_loader():
    """Test function to verify LibriLight loader functionality."""
    loader = LibriLightLoader("/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/")
    
    print(f"Total books discovered: {len(loader.books)}")
    print(f"Total speakers: {len(loader.speakers)}")
    
    # Test suitable books
    suitable_books = loader.get_suitable_books_for_ttt()
    print(f"Suitable books for TTT: {len(suitable_books)}")
    
    # Show a few examples
    for i, book in enumerate(suitable_books[:5]):
        print(f"{i+1}. {book}")
        print(f"   Estimated duration: {book.get_total_duration_estimate(3):.1f} minutes (3 chapters)")
    
    # Test random sequence creation
    result = loader.get_random_book_sequence(num_chapters=3)
    if result:
        book, chapters = result
        print(f"\nRandom sequence: '{book.title}' by {book.author}")
        print(f"Chapters: {[ch['name'] for ch in chapters]}")


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    test_librilight_loader()