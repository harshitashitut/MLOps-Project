"""
Unit tests for Interview Analyzer
Tests transcription and LLM output storage functionality
"""

import unittest
import os
import sys
from pathlib import Path
import shutil
import tempfile

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your InterviewAnalyzer class
from scripts.main2 import InterviewAnalyzer


class TestInterviewAnalyzer(unittest.TestCase):
    """Test cases for InterviewAnalyzer"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        print("\n" + "="*60)
        print("Setting up test environment...")
        print("="*60)
        
        # Create temporary storage directory for tests
        cls.test_storage_dir = tempfile.mkdtemp(prefix="test_store_")
        print(f"Test storage directory: {cls.test_storage_dir}")
        
        # Path to test video (adjust this to your actual test video path)
        cls.test_video_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "Data", 
            "video1.webm"
        )
        
        cls.test_question = "Tell me about yourself"
        
        print(f"Test video path: {cls.test_video_path}")
        print(f"Video exists: {os.path.exists(cls.test_video_path)}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print("\n" + "="*60)
        print("Cleaning up test environment...")
        print("="*60)
        
        # Remove temporary storage directory
        if os.path.exists(cls.test_storage_dir):
            shutil.rmtree(cls.test_storage_dir)
            print(f"Removed test storage directory: {cls.test_storage_dir}")
    
    def setUp(self):
        """Set up before each test"""
        # Check if InterviewAnalyzer supports storage_dir parameter
        import inspect
        sig = inspect.signature(InterviewAnalyzer.__init__)
        
        if 'storage_dir' in sig.parameters:
            # New version with storage_dir
            self.analyzer = InterviewAnalyzer(
                use_gpu=False,
                storage_dir=self.test_storage_dir
            )
        else:
            # Old version without storage_dir - skip tests that need it
            self.analyzer = InterviewAnalyzer(use_gpu=False)
            self.skipTest("InterviewAnalyzer doesn't support storage_dir parameter yet")
        
    def test_01_storage_directory_creation(self):
        """Test that storage directory is created"""
        print("\nTest 1: Storage directory creation")
        
        self.assertTrue(
            os.path.exists(self.test_storage_dir),
            "Storage directory should be created"
        )
        self.assertTrue(
            os.path.isdir(self.test_storage_dir),
            "Storage path should be a directory"
        )
        print("✓ Storage directory exists and is valid")
    
    def test_02_transcription_model_loading(self):
        """Test that transcription model loads successfully"""
        print("\nTest 2: Transcription model loading")
        
        self.analyzer.load_transcription_model("openai/whisper-base")
        
        self.assertIsNotNone(
            self.analyzer.transcription_pipeline,
            "Transcription pipeline should be loaded"
        )
        print("✓ Transcription model loaded successfully")
    
    def test_03_llm_model_loading(self):
        """Test that LLM model loads successfully"""
        print("\nTest 3: LLM model loading")
        
        # Use a small model for faster testing
        self.analyzer.load_llm_model("google/flan-t5-base")
        
        self.assertIsNotNone(
            self.analyzer.llm_pipeline,
            "LLM pipeline should be loaded"
        )
        print("✓ LLM model loaded successfully")
    
    @unittest.skipIf(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "Data", "video1.webm")),
        "Test video file not found"
    )
    def test_04_file_storage(self):
        """Test that analysis output is saved to file"""
        print("\nTest 4: File storage functionality")
        
        # Load models
        print("Loading models...")
        self.analyzer.load_transcription_model("openai/whisper-base")
        self.analyzer.load_llm_model("google/flan-t5-base")
        
        # Get initial file count
        initial_files = set(os.listdir(self.test_storage_dir))
        print(f"Initial files in storage: {len(initial_files)}")
        
        # Run analysis
        print("Running video analysis...")
        result = self.analyzer.analyze_video(
            self.test_video_path,
            self.test_question,
            save_analysis=True
        )
        
        # Check that a file was created
        final_files = set(os.listdir(self.test_storage_dir))
        new_files = final_files - initial_files
        
        self.assertEqual(
            len(new_files), 1,
            f"Exactly one new file should be created. Found: {len(new_files)}"
        )
        
        # Get the created file
        created_file = list(new_files)[0]
        file_path = os.path.join(self.test_storage_dir, created_file)
        
        print(f"✓ File created: {created_file}")
        
        # Verify file exists
        self.assertTrue(
            os.path.exists(file_path),
            "Analysis file should exist"
        )
        
        # Verify file is not empty
        file_size = os.path.getsize(file_path)
        self.assertGreater(
            file_size, 0,
            "Analysis file should not be empty"
        )
        print(f"✓ File size: {file_size} bytes")
        
        # Verify file contains expected sections
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            "INTERVIEW ANALYSIS REPORT",
            "QUESTION",
            "TRANSCRIPTION",
            "AI FEEDBACK"
        ]
        
        for section in required_sections:
            self.assertIn(
                section, content,
                f"File should contain '{section}' section"
            )
            print(f"✓ Found section: {section}")
        
        # Check that transcription is present
        self.assertIn(
            self.test_question, content,
            "File should contain the interview question"
        )
        print(f"✓ Question found in file")
        
        # Check that transcription has content (not empty)
        self.assertGreater(
            len(result['transcription']), 0,
            "Transcription should not be empty"
        )
        print(f"✓ Transcription length: {len(result['transcription'])} characters")
        
        # Check that feedback has content
        self.assertGreater(
            len(result['feedback']), 0,
            "Feedback should not be empty"
        )
        print(f"✓ Feedback length: {len(result['feedback'])} characters")
        
        # Verify the result contains analysis_file path
        self.assertIsNotNone(
            result.get('analysis_file'),
            "Result should contain analysis_file path"
        )
        print(f"✓ Analysis file path: {result['analysis_file']}")
    
    def test_05_save_analysis_function(self):
        """Test save_analysis method directly"""
        print("\nTest 5: Direct save_analysis method test")
        
        # Create mock data
        mock_transcription = "This is a test transcription."
        mock_feedback = "This is test feedback from the LLM."
        mock_video_path = "test_video.webm"
        mock_question = "What is your experience?"
        
        # Save analysis
        file_path = self.analyzer.save_analysis(
            transcription=mock_transcription,
            feedback=mock_feedback,
            video_path=mock_video_path,
            question=mock_question
        )
        
        # Verify file was created
        self.assertTrue(
            os.path.exists(file_path),
            "Analysis file should be created"
        )
        print(f"✓ File created at: {file_path}")
        
        # Read and verify content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn(mock_transcription, content, "File should contain transcription")
        self.assertIn(mock_feedback, content, "File should contain feedback")
        self.assertIn(mock_question, content, "File should contain question")
        
        print("✓ All expected content found in file")
    
    def test_06_storage_directory_parameter(self):
        """Test that custom storage directory is respected"""
        print("\nTest 6: Custom storage directory parameter")
        
        custom_dir = tempfile.mkdtemp(prefix="custom_store_")
        
        try:
            # Create analyzer with custom directory
            analyzer = InterviewAnalyzer(use_gpu=False, storage_dir=custom_dir)
            
            # Verify the directory was set correctly
            self.assertEqual(
                str(analyzer.storage_dir), custom_dir,
                "Storage directory should match the provided path"
            )
            
            # Verify directory exists
            self.assertTrue(
                os.path.exists(custom_dir),
                "Custom storage directory should be created"
            )
            
            print(f"✓ Custom storage directory: {custom_dir}")
            
        finally:
            # Clean up
            if os.path.exists(custom_dir):
                shutil.rmtree(custom_dir)
    
    def test_07_invalid_video_format(self):
        """Test that invalid video formats are rejected"""
        print("\nTest 7: Invalid video format validation")
        
        self.analyzer.load_transcription_model("openai/whisper-base")
        self.analyzer.load_llm_model("google/flan-t5-base")
        
        # Create a fake .txt file
        invalid_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        invalid_file.write(b"This is not a video file")
        invalid_file.close()
        
        try:
            # Should raise ValueError for invalid format
            with self.assertRaises(ValueError) as context:
                self.analyzer.analyze_video(invalid_file.name, "Test question")
            
            self.assertIn("Invalid video format", str(context.exception))
            print(f"✓ Correctly rejected invalid format: {context.exception}")
            
        finally:
            os.unlink(invalid_file.name)
    
    def test_08_missing_video_file(self):
        """Test that missing video files are handled"""
        print("\nTest 8: Missing video file validation")
        
        self.analyzer.load_transcription_model("openai/whisper-base")
        self.analyzer.load_llm_model("google/flan-t5-base")
        
        # Try to analyze non-existent file
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze_video("/fake/path/video.mp4", "Test question")
        
        self.assertIn("not found", str(context.exception).lower())
        print(f"✓ Correctly handled missing file: {context.exception}")
    
    def test_09_empty_question(self):
        """Test that empty questions are rejected"""
        print("\nTest 9: Empty question validation")
        
        self.analyzer.load_transcription_model("openai/whisper-base")
        self.analyzer.load_llm_model("google/flan-t5-base")
        
        # Create a valid video file path (even if fake, we'll catch it before ffmpeg)
        fake_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fake_video.write(b"fake video content")
        fake_video.close()
        
        try:
            # Test empty string
            with self.assertRaises(ValueError) as context:
                self.analyzer.analyze_video(fake_video.name, "")
            
            self.assertIn("question cannot be empty", str(context.exception))
            print(f"✓ Correctly rejected empty question")
            
            # Test whitespace only
            with self.assertRaises(ValueError) as context:
                self.analyzer.analyze_video(fake_video.name, "   ")
            
            self.assertIn("question cannot be empty", str(context.exception))
            print(f"✓ Correctly rejected whitespace-only question")
            
        finally:
            os.unlink(fake_video.name)
    
    def test_10_models_not_loaded(self):
        """Test that error is raised if models aren't loaded"""
        print("\nTest 10: Models not loaded validation")
        
        # Create valid dummy video file
        fake_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fake_video.write(b"fake video content")
        fake_video.close()
        
        try:
            # Try to analyze without loading models
            with self.assertRaises(RuntimeError) as context:
                self.analyzer.analyze_video(fake_video.name, "Test question")
            
            self.assertIn("not loaded", str(context.exception).lower())
            print(f"✓ Correctly detected missing models: {context.exception}")
            
        finally:
            os.unlink(fake_video.name)
    
    def test_11_empty_video_file(self):
        """Test that empty video files are rejected"""
        print("\nTest 11: Empty video file validation")
        
        self.analyzer.load_transcription_model("openai/whisper-base")
        self.analyzer.load_llm_model("google/flan-t5-base")
        
        # Create empty video file
        empty_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        empty_video.close()  # File exists but is empty
        
        try:
            with self.assertRaises(ValueError) as context:
                self.analyzer.analyze_video(empty_video.name, "Test question")
            
            self.assertIn("empty", str(context.exception).lower())
            print(f"✓ Correctly rejected empty file: {context.exception}")
            
        finally:
            os.unlink(empty_video.name)


def run_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInterviewAnalyzer)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    # Run tests
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)