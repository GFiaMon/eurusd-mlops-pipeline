"""
DataManager - Unified data access layer for EUR/USD ML Pipeline.

Provides Cloud-First, Local-Mirror architecture with:
- S3 as single source of truth
- Local mirroring in data/raw and data/processed
- Environment auto-detection (Lambda, EC2, local, offline)
- Smart bidirectional sync
- Support for both raw and processed data
"""

import os
import pandas as pd
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

# Optional imports (graceful degradation)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("boto3 not available - S3 functionality disabled")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning("joblib not available - scaler save/load disabled")

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified data access layer for EUR/USD ML pipeline.
    
    Examples:
        # Raw data access
        dm = DataManager(data_type='raw')
        df = dm.get_latest_data()
        
        # Processed data access
        dm = DataManager(data_type='processed')
        train, test, scaler = dm.load_processed()
        
        # Save new data
        dm.save_data(df, metadata={'source': 'local'})
    """
    
    def __init__(
        self,
        mode: str = 'auto',
        data_type: str = 'raw',
        local_dir: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        s3_region: str = 'us-east-1'
    ):
        """
        Initialize DataManager.
        
        Args:
            mode: 'auto', 'lambda', 'ec2', 'cloud', 'local', 'offline'
            data_type: 'raw' or 'processed'
            local_dir: Local mirror directory (default: data/raw or data/processed)
            s3_bucket: S3 bucket name (default: from env S3_BUCKET)
            s3_prefix: S3 prefix (default: data/raw/ or data/processed/)
            s3_region: AWS region (default: us-east-1)
        """
        # Configuration
        self.s3_region = s3_region or os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = s3_bucket or os.getenv('S3_BUCKET')
        
        # Environment detection
        self.mode = self._detect_environment() if mode == 'auto' else mode
        self.data_type = data_type
        
        # Local Directory Configuration
        if local_dir:
            self.local_dir = local_dir
        else:
            env_var = 'LOCAL_RAW_DIR' if data_type == 'raw' else 'LOCAL_PROCESSED_DIR'
            default = f'data/{data_type}'
            self.local_dir = os.getenv(env_var, default)
        
        if s3_prefix:
            self.s3_prefix = s3_prefix.rstrip('/') + '/'
        else:
            env_var = 'S3_RAW_PREFIX' if data_type == 'raw' else 'S3_PROCESSED_PREFIX'
            default = f'data/{data_type}/'
            self.s3_prefix = os.getenv(env_var, default)
        
        # Create local directory if needed (except in Lambda)
        
        # Create local directory if needed (except in Lambda)
        if self.mode != 'lambda':
            Path(self.local_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if in cloud mode
        self.s3_client = None
        if self.mode in ['lambda', 'ec2', 'cloud'] and S3_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3', region_name=self.s3_region)
                logger.info(f"DataManager: {self.mode} mode, {data_type} data, S3 enabled")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"S3 init failed: {e}. Falling back to local mode")
                self.mode = 'local'
        else:
            logger.info(f"DataManager: {self.mode} mode, {data_type} data, local only")
    
    def _detect_environment(self) -> str:
        """Auto-detect runtime environment."""
        # 1. Check for explicit override
        env_override = os.getenv('RUNTIME_ENV', '').lower()
        if env_override in ['lambda', 'ec2', 'cloud', 'local', 'offline']:
            return env_override
        
        # 2. Check for offline mode flag
        if os.getenv('FORCE_OFFLINE', 'false').lower() == 'true':
            return 'offline'
        
        # 3. Check for Lambda
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME') or os.getenv('AWS_EXECUTION_ENV'):
            return 'lambda'
        
        # 4. Check for EC2 (try to reach instance metadata)
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=0.1
            )
            if response.status_code == 200:
                return 'ec2'
        except:
            pass
        
        # 5. Check if cloud-capable (has S3 credentials)
        if os.getenv('S3_BUCKET') and S3_AVAILABLE:
            try:
                boto3.client('s3', region_name=self.s3_region)
                return 'cloud'
            except Exception as e:
                logger.warning(f"S3 detection failed: {e}")
                pass
        
        # 6. Default to local
        return 'local'
    
    # ==================== PUBLIC API ====================
    
    def is_cloud_available(self) -> bool:
        """Check if S3 access is available."""
        return self.s3_client is not None
    
    def get_environment(self) -> str:
        """Return current detected environment."""
        return self.mode
    
    def get_local_path(self, relative_path: str = '') -> str:
        """Get absolute path for a file in local directory."""
        return os.path.join(self.local_dir, relative_path)
    
    def get_latest_data(
        self,
        lookback_days: Optional[int] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get the latest EUR/USD data.
        
        Strategy:
        - Lambda/EC2: Load from S3 directly
        - Cloud: Sync from S3 to local, then load
        - Local: Load from local, optionally sync from S3
        - Offline: Load from local only
        
        Args:
            lookback_days: Number of days to retrieve (None = all)
            force_refresh: Force download from S3
            
        Returns:
            DataFrame with EUR/USD data, indexed by Date
        """
        logger.info(f"Getting latest {self.data_type} data (mode={self.mode}, force={force_refresh})")
        
        # Lambda/EC2: Read directly from S3
        if self.mode in ['lambda', 'ec2'] and self.is_cloud_available():
            df = self._load_from_s3()
        
        # Cloud/Local: Use local mirror with optional sync
        elif self.mode in ['cloud', 'local']:
            # Sync from S3 if available and (force or stale/missing)
            if self.is_cloud_available() and (force_refresh or self._is_local_stale()):
                logger.info("Syncing from S3...")
                self.sync_from_s3(force=force_refresh)
            
            # Load from local mirror
            df = self._load_from_local()
        
        # Offline: Local only
        else:
            logger.info("Offline mode: using local mirror only")
            df = self._load_from_local()
        
        # Filter by lookback_days if specified
        if lookback_days and not df.empty:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            df = df[df.index >= cutoff_date]
        
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save EUR/USD data.
        
        Strategy:
        - Lambda: Save directly to S3 with partition naming
        - Cloud/EC2/Local: Save to local + S3
        - Offline: Save to local only
        
        Args:
            df: DataFrame to save (must have Date index)
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        if df.empty:
            logger.warning("Cannot save empty DataFrame")
            return False
        
        # Ensure Date index
        if df.index.name != 'Date' and 'Date' not in df.columns:
            logger.error("DataFrame must have Date index or column")
            return False
        
        # Lambda: Direct S3 upload
        if self.mode == 'lambda':
            return self._save_to_s3_partitioned(df, metadata)
        
        # Flatten MultiIndex columns if present (yfinance style)
        # This ensures we save clean CSVs without the Price/Ticker/Date header mess
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Flattening MultiIndex columns")
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        
        # Cloud/EC2/Local: Save to local mirror
        if self.mode in ['cloud', 'ec2', 'local', 'offline']:
            success_local = self._save_to_local(df, metadata)
            
            # Also upload to S3 if available
            if self.is_cloud_available() and self.mode != 'offline':
                success_s3 = self._save_to_s3_partitioned(df, metadata)
                return success_local and success_s3
            
            return success_local
        
        return False
    
    def save_processed(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save processed data (train/test splits and scaler).
        
        Only works when data_type='processed'.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            scaler: Fitted scaler object
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        if self.data_type != 'processed':
            logger.error("save_processed() only works with data_type='processed'")
            return False
        
        if not JOBLIB_AVAILABLE:
            logger.error("joblib not available - cannot save scaler")
            return False
        
        try:
            # Save locally
            train_path = self.get_local_path('train.csv')
            test_path = self.get_local_path('test.csv')
            scaler_path = self.get_local_path('scaler.pkl')
            
            train_df.to_csv(train_path)
            test_df.to_csv(test_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Saved processed data to {self.local_dir}/")
            
            # Upload to S3 if available
            if self.is_cloud_available() and self.mode != 'offline':
                self.sync_to_s3(train_path)
                self.sync_to_s3(test_path)
                self.sync_to_s3(scaler_path)
                logger.info("Uploaded processed data to S3")
            
            # Save metadata
            if metadata:
                meta_path = self.get_local_path('metadata.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                if self.is_cloud_available():
                    self.sync_to_s3(meta_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False
    
    def load_processed(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[object]]:
        """
        Load processed data (train/test splits and scaler).
        
        Only works when data_type='processed'.
        
        Returns:
            Tuple of (train_df, test_df, scaler) or (None, None, None) if failed
        """
        if self.data_type != 'processed':
            logger.error("load_processed() only works with data_type='processed'")
            return None, None, None
        
        if not JOBLIB_AVAILABLE:
            logger.error("joblib not available - cannot load scaler")
            return None, None, None
        
        # Sync from S3 if needed
        if self.is_cloud_available() and self._is_local_stale():
            self.sync_from_s3()
        
        try:
            train_path = self.get_local_path('train.csv')
            test_path = self.get_local_path('test.csv')
            scaler_path = self.get_local_path('scaler.pkl')
            
            train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
            test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
            scaler = joblib.load(scaler_path)
            
            logger.info(f"Loaded processed data from {self.local_dir}/")
            return train_df, test_df, scaler
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return None, None, None
    
    def sync_from_s3(self, force: bool = False) -> bool:
        """
        Download data from S3 to local mirror.
        
        Args:
            force: Force download even if local is fresh
            
        Returns:
            True if successful
        """
        if not self.is_cloud_available():
            logger.warning("S3 not available, cannot sync")
            return False
        
        logger.info(f"Syncing from S3: s3://{self.s3_bucket}/{self.s3_prefix}")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            if 'Contents' not in response:
                logger.info("No files found in S3")
                return False
            
            # Download all files
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                
                if not filename:  # Skip directory markers
                    continue
                
                local_path = self.get_local_path(filename)
                
                # Download
                self.s3_client.download_file(self.s3_bucket, key, local_path)
                logger.debug(f"Downloaded {key} -> {local_path}")
            
            logger.info(f"Synced {len(response['Contents'])} files from S3")
            return True
            
        except Exception as e:
            logger.error(f"Sync from S3 failed: {e}")
            return False
    
    def sync_to_s3(self, local_path: str) -> bool:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Path to local file
            
        Returns:
            True if successful
        """
        if not self.is_cloud_available():
            logger.warning("S3 not available, cannot sync")
            return False
        
        try:
            filename = os.path.basename(local_path)
            s3_key = f"{self.s3_prefix}{filename}"
            
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Upload to S3 failed: {e}")
            return False
    
    def get_latest_date(self) -> Optional[datetime]:
        """Get the latest date in available data."""
        if self.is_cloud_available():
            return self._get_latest_date_from_s3()
        else:
            df = self._load_from_local()
            if not df.empty:
                return df.index.max()
        return None
    
    def is_data_current(self, max_age_hours: int = 24) -> bool:
        """Check if data is up-to-date (within max_age_hours)."""
        latest_date = self.get_latest_date()
        if not latest_date:
            return False
        
        # Convert to timezone-naive
        if hasattr(latest_date, 'tz') and latest_date.tz:
            latest_date = latest_date.replace(tzinfo=None)
        
        age = datetime.now() - latest_date
        return age < timedelta(hours=max_age_hours)
    
    # ==================== INTERNAL METHODS ====================
    
    def _load_from_s3(self) -> pd.DataFrame:
        """Load and merge all partitioned CSV files from S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            if 'Contents' not in response:
                return pd.DataFrame()
            
            dfs = []
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.csv'):
                    # Download and read
                    obj_data = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                    df_temp = pd.read_csv(
                        obj_data['Body'],
                        index_col=0,
                        parse_dates=True,
                        skiprows=[1, 2]  # Skip yfinance metadata
                    )
                    dfs.append(df_temp)
            
            if not dfs:
                return pd.DataFrame()
            
            # Merge and deduplicate
            df = pd.concat(dfs, axis=0)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from S3: {e}")
            return pd.DataFrame()
    
    def _load_from_local(self) -> pd.DataFrame:
        """Load data from local mirror."""
        # For raw data, look for merged file or individual files
        if self.data_type == 'raw':
            # Helper to safely load potentially messy yfinance CSVs
            def load_safe(filepath):
                # Peek at file to check for yfinance 3-line header
                try:
                    with open(filepath, 'r') as f:
                        lines = [f.readline() for _ in range(3)]
                    
                    # Check for Ticker line (standard yf output)
                    if len(lines) >= 2 and 'Ticker' in lines[1]:
                        return pd.read_csv(filepath, index_col=0, parse_dates=True, skiprows=[1, 2])
                    
                    # Standard load
                    return pd.read_csv(filepath, index_col=0, parse_dates=True)
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
                    raise e

            # Try merged file first
            merged_file = self.get_local_path('eurusd_latest.csv')
            if os.path.exists(merged_file):
                try:
                    df = load_safe(merged_file)
                    logger.info(f"Loaded {len(df)} rows from {merged_file}")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load {merged_file}: {e}")
            
            # Try legacy file
            legacy_file = self.get_local_path('eurusd_raw.csv')
            if os.path.exists(legacy_file):
                try:
                    df = load_safe(legacy_file)
                    logger.info(f"Loaded {len(df)} rows from {legacy_file}")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load {legacy_file}: {e}")
            
            # Try partitioned files
            csv_files = list(Path(self.local_dir).glob('data_*.csv'))
            if csv_files:
                dfs = []
                for csv_file in csv_files:
                    try:
                        df_temp = load_safe(csv_file)
                        dfs.append(df_temp)
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_file}: {e}")
                
                if dfs:
                    df = pd.concat(dfs, axis=0)
                    df = df.sort_index()
                    df = df[~df.index.duplicated(keep='last')]
                    logger.info(f"Loaded {len(df)} rows from {len(dfs)} partitioned files")
                    return df
        
        logger.warning(f"No data found in {self.local_dir}/")
        return pd.DataFrame()
    
    def _save_to_local(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """Save data to local mirror."""
        try:
            # Flatten MultiIndex before saving (consistent with S3 save)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            
            # For raw data, save as merged file
            if self.data_type == 'raw':
                local_file = self.get_local_path('eurusd_latest.csv')
            else:
                local_file = self.get_local_path('data.csv')
            
            df.to_csv(local_file)
            logger.info(f"Saved {len(df)} rows to {local_file}")
            
            # Save metadata
            if metadata:
                meta_file = self.get_local_path('metadata.json')
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to local: {e}")
            return False
    
    
    def _save_to_s3_partitioned(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """Save data to S3 with partitioned naming."""
        try:
            # Flatten MultiIndex before saving if needed (e.g. Lambda)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
                
            # Determine date range
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            
            # Create filename
            filename = f"data_{end_date}_from_{start_date}.csv"
            
            # For Lambda, use /tmp
            if self.mode == 'lambda':
                local_path = f"/tmp/{filename}"
            else:
                local_path = self.get_local_path(filename)
            
            # Save locally first
            df.to_csv(local_path)
            
            # Upload to S3
            s3_key = f"{self.s3_prefix}{filename}"
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            
            logger.info(f"Saved to S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to S3: {e}")
            return False
    
    def _is_local_stale(self, max_age_hours: int = 24) -> bool:
        """Check if local mirror is stale."""
        # Check for any CSV file in local directory
        csv_files = list(Path(self.local_dir).glob('*.csv'))
        
        if not csv_files:
            return True
        
        # Check modification time of most recent file
        latest_mtime = max(f.stat().st_mtime for f in csv_files)
        age = datetime.now().timestamp() - latest_mtime
        
        return age > (max_age_hours * 3600)
    
    def _get_latest_date_from_s3(self) -> Optional[datetime]:
        """Get the latest date from S3 files."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            if 'Contents' not in response:
                return None
            
            max_date = None
            pattern = re.compile(r"data_(\d{4}-\d{2}-\d{2})_from_(\d{4}-\d{2}-\d{2})\.csv")
            
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                match = pattern.match(filename)
                if match:
                    end_date_str = match.group(1)
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                    if max_date is None or end_date > max_date:
                        max_date = end_date
            
            return max_date
            
        except Exception as e:
            logger.error(f"Error getting latest date from S3: {e}")
            return None
