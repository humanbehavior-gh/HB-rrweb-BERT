#!/usr/bin/env python3
"""
High-performance RRWEB data downloader with concurrent downloads
Downloads RRWEB session data files with configurable concurrency
"""

import os
import json
import time
import uuid
import random
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import logging
from datetime import datetime
from tqdm.asyncio import tqdm
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_rrweb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RRWebDownloader:
    """High-performance RRWEB data downloader"""
    
    def __init__(self, 
                 output_dir: str,
                 num_files: int = 200000,
                 max_concurrent: int = 1000,
                 api_endpoint: str = None,
                 retry_attempts: int = 3,
                 timeout: int = 30):
        """
        Initialize the downloader
        
        Args:
            output_dir: Directory to save downloaded files
            num_files: Total number of files to download
            max_concurrent: Maximum number of concurrent downloads
            api_endpoint: API endpoint for downloading RRWEB data
            retry_attempts: Number of retry attempts for failed downloads
            timeout: Timeout for each download in seconds
        """
        self.output_dir = Path(output_dir)
        self.num_files = num_files
        self.max_concurrent = max_concurrent
        self.api_endpoint = api_endpoint or self._generate_mock_endpoint()
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.downloaded = 0
        self.failed = 0
        self.start_time = None
        
        # Graceful shutdown
        self.shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown = True
    
    def _generate_mock_endpoint(self):
        """Generate mock endpoint for testing"""
        # In production, replace with actual RRWEB data endpoint
        return "https://api.example.com/rrweb/sessions"
    
    def _generate_mock_rrweb_data(self) -> Dict[str, Any]:
        """Generate mock RRWEB session data for testing"""
        # This creates realistic-looking RRWEB data structure
        session_id = str(uuid.uuid4())
        events = []
        
        # Add some mock events
        for i in range(random.randint(50, 500)):
            event = {
                "type": random.choice([2, 3, 4, 5, 6]),
                "data": {
                    "source": random.randint(0, 10),
                    "x": random.randint(0, 1920),
                    "y": random.randint(0, 1080),
                    "timestamp": i * 100
                },
                "timestamp": i * 100
            }
            
            # Add some DOM snapshot data for type 2 events
            if event["type"] == 2:
                event["data"]["node"] = {
                    "type": 2,
                    "tagName": random.choice(["div", "button", "input", "form", "span"]),
                    "attributes": {
                        "class": f"class-{random.randint(1, 100)}",
                        "id": f"id-{random.randint(1, 50)}"
                    },
                    "textContent": f"Sample text {random.randint(1, 100)}"
                }
            
            events.append(event)
        
        return {
            "session_id": session_id,
            "duration": len(events) * 100 / 1000,  # Duration in seconds
            "events": events,
            "metadata": {
                "url": f"https://example.com/page{random.randint(1, 100)}",
                "timestamp": datetime.now().isoformat(),
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        }
    
    async def download_file(self, session: aiohttp.ClientSession, 
                           file_index: int, 
                           semaphore: asyncio.Semaphore) -> bool:
        """
        Download a single RRWEB file
        
        Args:
            session: aiohttp session for making requests
            file_index: Index of the file being downloaded
            semaphore: Semaphore for limiting concurrent downloads
            
        Returns:
            True if download successful, False otherwise
        """
        async with semaphore:
            if self.shutdown:
                return False
            
            file_id = str(uuid.uuid4())
            file_path = self.output_dir / f"{file_id}.json"
            
            for attempt in range(self.retry_attempts):
                try:
                    # In production, replace with actual API call
                    # async with session.get(
                    #     f"{self.api_endpoint}/{file_id}",
                    #     timeout=aiohttp.ClientTimeout(total=self.timeout)
                    # ) as response:
                    #     response.raise_for_status()
                    #     data = await response.json()
                    
                    # For now, generate mock data
                    data = self._generate_mock_rrweb_data()
                    
                    # Save to file
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(json.dumps(data, separators=(',', ':')))
                    
                    self.downloaded += 1
                    return True
                    
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        logger.error(f"Failed to download file {file_index} after {self.retry_attempts} attempts: {e}")
                        self.failed += 1
                        return False
                    else:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            return False
    
    async def download_batch(self, start_idx: int, end_idx: int):
        """
        Download a batch of files concurrently
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive)
        """
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            # Create download tasks
            tasks = []
            for i in range(start_idx, end_idx):
                if self.shutdown:
                    break
                task = self.download_file(session, i, semaphore)
                tasks.append(task)
            
            # Run downloads with progress bar
            desc = f"Downloading RRWEB files ({self.max_concurrent} threads)"
            for f in tqdm.as_completed(tasks, total=len(tasks), desc=desc):
                await f
                if self.shutdown:
                    break
    
    async def run(self):
        """Run the downloader"""
        self.start_time = time.time()
        logger.info(f"Starting download of {self.num_files} RRWEB files to {self.output_dir}")
        logger.info(f"Using {self.max_concurrent} concurrent connections")
        
        # Download in batches to avoid overwhelming memory
        batch_size = min(10000, self.num_files)
        
        for i in range(0, self.num_files, batch_size):
            if self.shutdown:
                break
            
            batch_end = min(i + batch_size, self.num_files)
            logger.info(f"Downloading batch {i//batch_size + 1}: files {i}-{batch_end}")
            
            await self.download_batch(i, batch_end)
            
            # Log progress
            elapsed = time.time() - self.start_time
            rate = self.downloaded / elapsed if elapsed > 0 else 0
            eta = (self.num_files - self.downloaded) / rate if rate > 0 else 0
            
            logger.info(f"Progress: {self.downloaded}/{self.num_files} files "
                       f"({self.downloaded/self.num_files*100:.1f}%), "
                       f"Failed: {self.failed}, "
                       f"Rate: {rate:.1f} files/sec, "
                       f"ETA: {eta/60:.1f} minutes")
        
        # Final statistics
        elapsed = time.time() - self.start_time
        logger.info(f"\nDownload completed!")
        logger.info(f"Total files downloaded: {self.downloaded}")
        logger.info(f"Failed downloads: {self.failed}")
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Average rate: {self.downloaded/elapsed:.2f} files/second")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Download RRWEB session data')
    parser.add_argument('--output-dir', '-o', 
                       default='/home/ubuntu/rrweb-bert/rrweb_data',
                       help='Output directory for downloaded files')
    parser.add_argument('--num-files', '-n', 
                       type=int, default=200000,
                       help='Number of files to download')
    parser.add_argument('--concurrent', '-c', 
                       type=int, default=1000,
                       help='Maximum concurrent downloads')
    parser.add_argument('--api-endpoint', '-a',
                       help='API endpoint for RRWEB data')
    parser.add_argument('--retry-attempts', '-r',
                       type=int, default=3,
                       help='Number of retry attempts for failed downloads')
    parser.add_argument('--timeout', '-t',
                       type=int, default=30,
                       help='Timeout per download in seconds')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = RRWebDownloader(
        output_dir=args.output_dir,
        num_files=args.num_files,
        max_concurrent=args.concurrent,
        api_endpoint=args.api_endpoint,
        retry_attempts=args.retry_attempts,
        timeout=args.timeout
    )
    
    # Run async downloader
    try:
        asyncio.run(downloader.run())
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()