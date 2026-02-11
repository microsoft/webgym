import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

class FontManager:
    """Manages font downloading and caching for Unicode support."""
    
    # Font URLs from Google Fonts GitHub releases and other sources
    FONT_URLS = {
        "NotoSans-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        "NotoSansCJKsc-Regular.otf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        "NotoSansCJKjp-Regular.otf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf", 
        "NotoSansCJKkr-Regular.otf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf",
        "NotoSansArabic-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf",
        "NotoSansDevanagari-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
        "NotoSansSymbols2-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf",
        "NotoColorEmoji.ttf": "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf",
    }
    
    # Alternative URLs for some fonts (fallback)
    ALTERNATIVE_URLS = {
        "NotoSans-Regular.ttf": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        "NotoColorEmoji.ttf": "https://github.com/googlefonts/noto-emoji/releases/latest/download/NotoColorEmoji.ttf",
    }
    
    # Expected file sizes (approximate, in bytes) for validation
    EXPECTED_SIZES = {
        "NotoSans-Regular.ttf": (300000, 600000),  # 300KB - 600KB
        "NotoSansCJKsc-Regular.otf": (15000000, 20000000),  # 15MB - 20MB
        "NotoSansCJKjp-Regular.otf": (15000000, 20000000),  # 15MB - 20MB
        "NotoSansCJKkr-Regular.otf": (15000000, 20000000),  # 15MB - 20MB
        "NotoSansArabic-Regular.ttf": (100000, 300000),  # 100KB - 300KB
        "NotoSansDevanagari-Regular.ttf": (150000, 400000),  # 150KB - 400KB
        "NotoSansSymbols2-Regular.ttf": (800000, 1500000),  # 800KB - 1.5MB
        "NotoColorEmoji.ttf": (8000000, 15000000),  # 8MB - 15MB
    }
    
    def __init__(self, font_directory: str = None):
        """
        Initialize FontManager with a directory for storing fonts.
        
        Args:
            font_directory: Directory to store/load fonts. If None, uses ~/.fonts/noto
        """
        if font_directory is None:
            font_directory = os.path.expanduser("~/.fonts/noto")
        
        self.font_directory = Path(font_directory)
        self.font_directory.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded fonts
        self.font_paths: Dict[str, Path] = {}
        
    def download_font(self, font_name: str, url: str, retry_with_alt: bool = True) -> bool:
        """
        Download a single font file.
        
        Args:
            font_name: Name of the font file
            url: URL to download from
            retry_with_alt: Whether to retry with alternative URL on failure
            
        Returns:
            True if download successful, False otherwise
        """
        font_path = self.font_directory / font_name
        
        try:
            print(f"Downloading {font_name} from {url}...")
            
            # Download with streaming to handle large files
            response = requests.get(url, stream=True, timeout=30, 
                                   headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            # Write to temporary file first
            temp_path = font_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Validate file size if expected size is known
            if font_name in self.EXPECTED_SIZES:
                file_size = temp_path.stat().st_size
                min_size, max_size = self.EXPECTED_SIZES[font_name]
                if not (min_size <= file_size <= max_size):
                    print(f"Warning: {font_name} size {file_size} bytes is outside expected range {min_size}-{max_size}")
                    # Continue anyway, just warn
            
            # Move to final location
            temp_path.rename(font_path)
            print(f"Successfully downloaded {font_name}")
            return True
            
        except Exception as e:
            print(f"Failed to download {font_name} from {url}: {e}")
            
            # Try alternative URL if available
            if retry_with_alt and font_name in self.ALTERNATIVE_URLS:
                alt_url = self.ALTERNATIVE_URLS[font_name]
                print(f"Trying alternative URL: {alt_url}")
                return self.download_font(font_name, alt_url, retry_with_alt=False)
            
            return False
    
    def ensure_font(self, font_name: str) -> Optional[Path]:
        """
        Ensure a font exists, downloading if necessary.
        
        Args:
            font_name: Name of the font file
            
        Returns:
            Path to the font file if successful, None otherwise
        """
        font_path = self.font_directory / font_name
        
        # Check if already exists
        if font_path.exists():
            print(f"Font {font_name} already exists at {font_path}")
            return font_path
        
        # Try to download
        if font_name in self.FONT_URLS:
            if self.download_font(font_name, self.FONT_URLS[font_name]):
                return font_path
        else:
            print(f"Unknown font: {font_name}")
        
        return None
    
    def ensure_all_fonts(self, fonts: List[str] = None) -> Dict[str, Path]:
        """
        Ensure all specified fonts exist, downloading as needed.
        
        Args:
            fonts: List of font names to ensure. If None, ensures all known fonts.
            
        Returns:
            Dictionary mapping font names to their paths (None if failed)
        """
        if fonts is None:
            fonts = list(self.FONT_URLS.keys())
        
        result = {}
        for font_name in fonts:
            result[font_name] = self.ensure_font(font_name)
        
        return result
    
    def get_font_categories(self) -> Dict[str, List[Path]]:
        """
        Get fonts organized by category after ensuring they exist.
        
        Returns:
            Dictionary with categories as keys and lists of font paths as values
        """
        # Define which fonts belong to which categories
        categories = {
            "base": ["NotoSans-Regular.ttf"],
            "cjk_sc": ["NotoSansCJKsc-Regular.otf"],
            "cjk_jp": ["NotoSansCJKjp-Regular.otf"],
            "cjk_kr": ["NotoSansCJKkr-Regular.otf"],
            "arabic": ["NotoSansArabic-Regular.ttf"],
            "devanagari": ["NotoSansDevanagari-Regular.ttf"],
            "symbols2": ["NotoSansSymbols2-Regular.ttf"],
            "emoji": ["NotoColorEmoji.ttf"],
        }
        
        result = {}
        for category, font_names in categories.items():
            paths = []
            for font_name in font_names:
                font_path = self.ensure_font(font_name)
                if font_path:
                    paths.append(font_path)
            if paths:
                result[category] = paths
        
        return result
    
    def cleanup_temp_files(self):
        """Remove any temporary files from failed downloads."""
        for tmp_file in self.font_directory.glob("*.tmp"):
            try:
                tmp_file.unlink()
                print(f"Removed temporary file: {tmp_file}")
            except Exception as e:
                print(f"Failed to remove {tmp_file}: {e}")


# Integration function for your existing code
def setup_fonts_with_auto_download(font_directory: str = None, font_size: int = 28):
    """
    Set up fonts with automatic downloading of missing fonts.
    
    Args:
        font_directory: Directory to store/load fonts. If None, uses ~/.fonts/noto
        font_size: Font size to use
        
    Returns:
        List of PIL ImageFont objects ready for use
    """
    from PIL import ImageFont
    
    # Initialize font manager
    manager = FontManager(font_directory)
    
    # Get all font paths organized by category
    font_categories = manager.get_font_categories()
    
    # Create ImageFont objects
    fonts = []
    for category in ["base", "cjk_sc", "cjk_jp", "cjk_kr", "arabic", 
                     "devanagari", "symbols2", "emoji"]:
        if category in font_categories:
            for font_path in font_categories[category]:
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                    fonts.append(font)
                    print(f"Loaded font: {font_path}")
                except Exception as e:
                    print(f"Failed to load font {font_path}: {e}")
    
    # Clean up any temp files
    manager.cleanup_temp_files()
    
    return fonts


# Example usage to replace your existing font setup
if __name__ == "__main__":
    # Example: Download fonts to a custom directory
    custom_font_dir = "/logs/fonts"  # Change this to your desired directory
    
    # This will download all missing fonts and return loaded ImageFont objects
    loaded_fonts = setup_fonts_with_auto_download(
        font_directory=custom_font_dir,
        font_size=28
    )
    
    print(f"\nSuccessfully loaded {len(loaded_fonts)} fonts")
    
    # You can also use the FontManager directly for more control
    manager = FontManager(custom_font_dir)
    
    # Download specific fonts
    manager.ensure_font("NotoSans-Regular.ttf")
    manager.ensure_font("NotoColorEmoji.ttf")
    
    # Or download all known fonts
    all_fonts = manager.ensure_all_fonts()
    for font_name, font_path in all_fonts.items():
        if font_path:
            print(f"{font_name}: {font_path}")
        else:
            print(f"{font_name}: Failed to download")