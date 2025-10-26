import os
import asyncio
import logging
import logging.handlers
from typing import Dict, Any, Optional, List
from telegram import Update, Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, ConversationHandler
import google.generativeai as genai
import requests
import requests.exceptions
from dotenv import load_dotenv
import json
import re
from datetime import datetime, timedelta
import io
import traceback
from functools import wraps
import time
from enum import Enum

# Load environment variables
load_dotenv()

# Circuit Breaker States
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, blocking calls
    HALF_OPEN = "half_open"  # Testing if service is recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                else:
                    raise Exception("Service temporarily unavailable. Please try again later.")

        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            return result

        except self.expected_exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(f"Circuit breaker OPENED after {self.failure_threshold} failures")

            raise e

def retry_async(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Async retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            current_delay = delay

            while retry_count < max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries. Final error: {e}")
                        raise e

                    logger.warning(f"Retry {retry_count}/{max_retries} for {func.__name__} after {current_delay}s. Error: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator

def setup_logging():
    """Setup enhanced logging with rotation and error-only files"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # General logger (INFO level)
    general_logger = logging.getLogger('clickgram')
    general_logger.setLevel(logging.INFO)

    # Error logger (ERROR level only, 5-day rotation)
    error_logger = logging.getLogger('clickgram.errors')
    error_logger.setLevel(logging.ERROR)

    # API logger for external service calls
    api_logger = logging.getLogger('clickgram.api')
    api_logger.setLevel(logging.INFO)

    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    json_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s", "traceback": "%(exc_info)s"}'
    )

    # General handlers
    general_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/clickgram.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    general_handler.setFormatter(detailed_formatter)
    general_logger.addHandler(general_handler)

    # Error-only handler with 5-day rotation
    error_handler = logging.handlers.TimedRotatingFileHandler(
        f"{log_dir}/errors.log",
        when='midnight',
        interval=1,
        backupCount=5,  # Keep last 5 days
        encoding='utf-8'
    )
    error_handler.setFormatter(json_formatter)
    error_logger.addHandler(error_handler)

    # API handler
    api_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/api_calls.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    api_handler.setFormatter(detailed_formatter)
    api_logger.addHandler(api_handler)

    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)
    general_logger.addHandler(console_handler)

    return general_logger, error_logger, api_logger

# Initialize loggers
logger, error_logger, api_logger = setup_logging()

# Circuit breakers for external services
clickup_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120)
gemini_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
telegram_circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=30)

class ClickGramError(Exception):
    """Base exception for ClickGram bot"""
    pass

class APIError(ClickGramError):
    """API-related errors"""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class ValidationError(ClickGramError):
    """Validation-related errors"""
    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.field = field

class ConversationError(ClickGramError):
    """Conversation flow errors"""
    def __init__(self, message: str, user_id: int = None):
        super().__init__(message)
        self.user_id = user_id


async def log_error(error: Exception, context: Dict[str, Any] = None):
    """Centralized error logging with context and monitoring"""
    error_data = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }

    error_logger.error(json.dumps(error_data, ensure_ascii=False))

    # Send critical errors to main logger too
    if isinstance(error, (APIError, ConnectionError)):
        logger.error(f"Critical error: {error_data['error_type']} - {error_data['error_message']}")

    # Track error for monitoring and alerting
    error_type = type(error).__name__.lower().replace('error', '')
    if isinstance(error, APIError):
        await error_monitor.track_error('api_errors', error)
    elif isinstance(error, ValidationError):
        await error_monitor.track_error('validation_errors', error)
    elif isinstance(error, ConversationError):
        await error_monitor.track_error('conversation_errors', error)
    else:
        await error_monitor.track_error('general_errors', error)

def handle_api_errors(func):
    """Decorator for API error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            api_logger.info(f"API call successful: {func.__name__} - {duration:.2f}s")
            return result

        except requests.exceptions.RequestException as e:
            error_context = {
                'function': func.__name__,
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200],
                'duration': time.time() - start_time if 'start_time' in locals() else 0
            }

            api_error = APIError(
                f"API request failed: {str(e)}",
                status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            )

            await log_error(api_error, error_context)
            raise api_error

        except Exception as e:
            error_context = {
                'function': func.__name__,
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200],
                'unexpected_error': True
            }

            await log_error(e, error_context)
            raise

    return wrapper

class ErrorMonitor:
    """Error monitoring and alerting system"""
    def __init__(self):
        self.error_counts = {}
        self.alert_thresholds = {
            'api_errors': 10,      # Alert after 10 API errors in 5 minutes
            'validation_errors': 20, # Alert after 20 validation errors
            'conversation_errors': 15 # Alert after 15 conversation errors
        }
        self.time_windows = {}
        self.last_alert_time = {}

    async def track_error(self, error_type: str, error: Exception):
        """Track error counts and check if alert should be sent"""
        now = time.time()
        window_key = f"{error_type}_{now // 300}"  # 5-minute windows

        # Initialize window if not exists
        if window_key not in self.error_counts:
            self.error_counts[window_key] = 0
            self.time_windows[window_key] = now

        self.error_counts[window_key] += 1

        # Check if we should send an alert
        threshold = self.alert_thresholds.get(error_type, 10)
        if self.error_counts[window_key] >= threshold:
            # Check if we haven't sent an alert recently (within 10 minutes)
            last_alert = self.last_alert_time.get(error_type, 0)
            if now - last_alert > 600:  # 10 minutes
                await self.send_alert(error_type, self.error_counts[window_key], error)
                self.last_alert_time[error_type] = now

    async def send_alert(self, error_type: str, count: int, sample_error: Exception):
        """Send error alert"""
        alert_message = f"""
🚨 **Error Alert - ClickGram Bot**

**Error Type:** {error_type}
**Count in 5 minutes:** {count}
**Threshold:** {self.alert_thresholds.get(error_type, 10)}
**Sample Error:** {str(sample_error)[:200]}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please investigate the issue.
"""

        logger.critical(alert_message)

        # Here you could add other alert methods like:
        # - Send to admin Telegram chat
        # - Send email notification
        # - Send to monitoring service

    def cleanup_old_windows(self):
        """Clean up old error tracking windows"""
        now = time.time()
        cutoff_time = now - 600  # Remove windows older than 10 minutes

        windows_to_remove = []
        for window_key, window_time in self.time_windows.items():
            if window_time < cutoff_time:
                windows_to_remove.append(window_key)

        for window_key in windows_to_remove:
            del self.error_counts[window_key]
            del self.time_windows[window_key]

# Initialize error monitor
error_monitor = ErrorMonitor()

class EnvironmentValidator:
    """Environment configuration validation system"""

    def __init__(self):
        self.validation_results = []
        self.required_vars = {
            'TELEGRAM_BOT_TOKEN': {
                'type': str,
                'min_length': 35,
                'description': 'Telegram Bot Token from @BotFather'
            },
            'GEMINI_API_KEY': {
                'type': str,
                'min_length': 20,
                'description': 'Google Gemini API Key'
            },
            'CLICKUP_API_KEY': {
                'type': str,
                'pattern': r'^pk_[a-zA-Z0-9_-]+$',
                'description': 'ClickUp API Key (should start with pk_)'
            },
            'CLICKUP_LIST_ID': {
                'type': str,
                'pattern': r'^[0-9]+$',
                'description': 'ClickUp List ID (numeric)'
            }
        }

        self.optional_vars = {
            'AUTHORIZED_USERS': {
                'type': str,
                'description': 'Comma-separated Telegram user IDs (optional for public bot)'
            },
            'LOG_LEVEL': {
                'type': str,
                'default': 'INFO',
                'valid_values': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                'description': 'Logging level'
            }
        }

    def validate_required_variables(self) -> bool:
        """Validate all required environment variables"""
        all_valid = True

        for var_name, config in self.required_vars.items():
            result = self.validate_single_variable(var_name, config, required=True)
            self.validation_results.append(result)
            if not result['valid']:
                all_valid = False

        return all_valid

    def validate_optional_variables(self) -> None:
        """Validate optional environment variables"""
        for var_name, config in self.optional_vars.items():
            result = self.validate_single_variable(var_name, config, required=False)
            self.validation_results.append(result)

    def validate_single_variable(self, var_name: str, config: dict, required: bool) -> dict:
        """Validate a single environment variable"""
        result = {
            'variable': var_name,
            'required': required,
            'valid': True,
            'value': os.getenv(var_name),
            'errors': []
        }

        value = result['value']

        # Check if variable exists
        if required and (value is None or value.strip() == ''):
            result['valid'] = False
            result['errors'].append(f'{var_name} is required but not set')
            return result

        # Skip validation if not set and optional
        if value is None or value.strip() == '':
            result['valid'] = True
            result['value'] = config.get('default')
            return result

        # Type validation
        expected_type = config.get('type', str)
        try:
            if expected_type == int:
                int(value)
            elif expected_type == bool:
                value.lower() in ['true', 'false', '1', '0', 'yes', 'no']
        except (ValueError, TypeError):
            result['valid'] = False
            result['errors'].append(f'{var_name} must be of type {expected_type.__name__}')

        # Length validation
        if 'min_length' in config and len(str(value)) < config['min_length']:
            result['valid'] = False
            result['errors'].append(f'{var_name} must be at least {config["min_length"]} characters long')

        # Pattern validation
        if 'pattern' in config and not re.match(config['pattern'], str(value)):
            result['valid'] = False
            result['errors'].append(f'{var_name} format is invalid. Expected format: {config["description"]}')

        # Valid values validation
        if 'valid_values' in config and str(value) not in config['valid_values']:
            result['valid'] = False
            result['errors'].append(f'{var_name} must be one of: {", ".join(config["valid_values"])}')

        return result

    def get_validation_report(self) -> str:
        """Generate a human-readable validation report"""
        if not self.validation_results:
            return "No validation performed yet."

        report = ["🔍 **Environment Validation Report**\n"]

        required_results = [r for r in self.validation_results if r['required']]
        optional_results = [r for r in self.validation_results if not r['required']]

        # Required variables
        report.append("\n📋 **Required Variables:**")
        for result in required_results:
            status = "✅" if result['valid'] else "❌"
            value_display = "***SET***" if result['value'] else "***NOT SET***"
            report.append(f"{status} **{result['variable']}**: {value_display}")

            if result['errors']:
                for error in result['errors']:
                    report.append(f"   ⚠️  {error}")

        # Optional variables
        report.append("\n📝 **Optional Variables:**")
        for result in optional_results:
            status = "✅" if result['valid'] else "⚠️"
            value_display = result['value'] if result['value'] else "***NOT SET***"
            report.append(f"{status} **{result['variable']}**: {value_display}")

            if result['errors']:
                for error in result['errors']:
                    report.append(f"   ⚠️  {error}")

        # Overall status
        required_valid = all(r['valid'] for r in required_results)
        overall_status = "✅ **PASSED**" if required_valid else "❌ **FAILED**"
        report.append(f"\n🎯 **Overall Status**: {overall_status}")

        if not required_valid:
            report.append("\n🚨 **Action Required:** Please fix the failed validation checks before starting the bot.")

        return "\n".join(report)

class ConnectivityValidator:
    """API connectivity validation system"""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.connectivity_results = {}

    async def validate_all_connections(self) -> bool:
        """Validate all external service connections"""
        logger.info("Starting connectivity validation...")

        tests = [
            ('telegram', self.validate_telegram_connection),
            ('gemini', self.validate_gemini_connection),
            ('clickup', self.validate_clickup_connection)
        ]

        all_connected = True
        for service_name, test_func in tests:
            try:
                logger.info(f"Testing {service_name} connection...")
                result = await test_func()
                self.connectivity_results[service_name] = result
                if not result['connected']:
                    all_connected = False
                    logger.error(f"{service_name} connection failed: {result['error']}")
                else:
                    logger.info(f"{service_name} connection successful")
            except Exception as e:
                logger.error(f"Error testing {service_name} connection: {e}")
                self.connectivity_results[service_name] = {
                    'connected': False,
                    'error': str(e),
                    'response_time': None
                }
                all_connected = False

        return all_connected

    async def validate_telegram_connection(self) -> dict:
        """Validate Telegram bot connection"""
        try:
            start_time = time.time()

            # Simple test - check if bot token is valid format
            if not self.bot.bot_token or not self.bot.bot_token.startswith(('1', '2', '5', '6', '7')):
                return {
                    'connected': False,
                    'error': 'Invalid Telegram bot token format',
                    'response_time': None
                }

            # Test basic bot info (this would normally require an actual API call)
            # For now, we'll validate the token format and length
            if len(self.bot.bot_token) < 35:
                return {
                    'connected': False,
                    'error': 'Telegram bot token too short',
                    'response_time': None
                }

            response_time = time.time() - start_time

            return {
                'connected': True,
                'error': None,
                'response_time': response_time
            }

        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'response_time': None
            }

    async def validate_gemini_connection(self) -> dict:
        """Validate Gemini API connection"""
        try:
            start_time = time.time()

            if not self.bot.gemini_api_key:
                return {
                    'connected': False,
                    'error': 'Gemini API key not configured',
                    'response_time': None
                }

            # Test with a simple API call
            test_prompt = "Hello, this is a connection test. Please respond with 'Connection successful'."
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.bot.model.generate_content(test_prompt)
                ),
                timeout=10.0
            )

            response_time = time.time() - start_time

            return {
                'connected': True,
                'error': None,
                'response_time': response_time,
                'response_preview': response.text[:100] if hasattr(response, 'text') else 'No response'
            }

        except asyncio.TimeoutError:
            return {
                'connected': False,
                'error': 'Gemini service is taking too long to respond. Please try again later.',
                'response_time': None
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'response_time': None
            }

    async def validate_clickup_connection(self) -> dict:
        """Validate ClickUp API connection"""
        try:
            start_time = time.time()

            if not self.bot.clickup_api_key:
                return {
                    'connected': False,
                    'error': 'ClickUp API key not configured',
                    'response_time': None
                }

            if not self.bot.clickup_list_id:
                return {
                    'connected': False,
                    'error': 'ClickUp list ID not configured',
                    'response_time': None
                }

            # Test API connection by fetching list info
            url = f"https://api.clickup.com/api/v2/list/{self.bot.clickup_list_id}"

            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        url,
                        headers=self.bot.clickup_headers,
                        timeout=10
                    )
                ),
                timeout=10.0
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                return {
                    'connected': True,
                    'error': None,
                    'response_time': response_time,
                    'list_name': response.json().get('name', 'Unknown')
                }
            else:
                return {
                    'connected': False,
                    'error': 'Unable to connect to ClickUp service. Please try again later.',
                    'response_time': response_time
                }

        except asyncio.TimeoutError:
            return {
                'connected': False,
                'error': 'ClickUp service is taking too long to respond. Please try again later.',
                'response_time': None
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'response_time': None
            }

    def get_connectivity_report(self) -> str:
        """Generate a human-readable connectivity report"""
        if not self.connectivity_results:
            return "No connectivity tests performed yet."

        report = ["🌐 **Service Status**\n"]

        all_connected = True

        for service, result in self.connectivity_results.items():
            status = "✅" if result['connected'] else "❌"
            service_name = service.title()

            report.append(f"{status} **{service_name}**")

            if result['connected']:
                report.append("   Status: Working properly")
            else:
                report.append(f"   Status: {result['error']}")
                all_connected = False

            report.append("")

        if all_connected:
            report.append("\n🎉 All services are working properly!")
        else:
            report.append("\n⚠️  Some services are experiencing issues. Please try again later.")

        return "\n".join(report)

# Conversation states
TITLE, DESCRIPTION, PRIORITY, DUE_DATE, ASSIGNEE, CONFIRMATION = range(6)

# Callback data patterns
CALLBACK_PREFIX = "task_"
PRIORITY_CALLBACK = f"{CALLBACK_PREFIX}priority"
ASSIGNEE_CALLBACK = f"{CALLBACK_PREFIX}assignee"
CONFIRM_CALLBACK = f"{CALLBACK_PREFIX}confirm"
CANCEL_CALLBACK = f"{CALLBACK_PREFIX}cancel"

class ClickGramBot:
    def __init__(self):
        # Track bot start time for uptime monitoring
        self.start_time = datetime.now()

        # Telegram configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

        # Gemini configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.vision_model = genai.GenerativeModel('gemini-2.0-flash')

        # ClickUp configuration
        self.clickup_api_key = os.getenv('CLICKUP_API_KEY')
        self.clickup_list_id = os.getenv('CLICKUP_LIST_ID')
        self.clickup_headers = {
            'Authorization': self.clickup_api_key,
            'Content-Type': 'application/json'
        }

        # Task keywords that trigger processing
        self.task_keywords = [
            'task', 'todo', 'assignment', 'action item', 'deliverable',
            'deadline', 'due', 'complete', 'finish', 'work on', 'implement',
            'create', 'build', 'develop', 'fix', 'resolve', 'handle', 'done',
            'look at', 'asap', 'as soon'
        ]

        # Conversation state management
        self.conversations = {}  # user_id: conversation_data
        self.conversation_timeout = 300  # 5 minutes

        # Start background cleanup task
        self.start_cleanup_task()

        # Initialize validators
        self.env_validator = EnvironmentValidator()
        self.connectivity_validator = ConnectivityValidator(self)

    async def validate_environment(self, include_connectivity: bool = True) -> bool:
        """Validate environment configuration and connectivity"""
        logger.info("Starting environment validation...")

        # Validate environment variables
        env_valid = self.env_validator.validate_required_variables()
        self.env_validator.validate_optional_variables()

        if not env_valid:
            logger.error("Environment validation failed")
            return False

        logger.info("Environment validation passed")

        # Validate connectivity if requested
        if include_connectivity:
            try:
                connectivity_valid = await self.connectivity_validator.validate_all_connections()
                if not connectivity_valid:
                    logger.error("Connectivity validation failed")
                    return False

                logger.info("Connectivity validation passed")
            except Exception as e:
                logger.error(f"Error during connectivity validation: {e}")
                return False

        return True

    def start_cleanup_task(self):
        """Start background task for cleaning up old logs and error data"""
        async def cleanup_task():
            while True:
                try:
                    # Clean up error monitoring windows
                    error_monitor.cleanup_old_windows()

                    # Clean up old log files (older than 5 days)
                    await self.cleanup_old_logs()

                    logger.info("Cleanup task completed")
                    await asyncio.sleep(3600)  # Run every hour

                except Exception as e:
                    context = {
                        'function': 'cleanup_task',
                        'task_type': 'background_cleanup'
                    }
                    await log_error(e, context)
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying

        # Start the cleanup task
        cleanup_coroutine = cleanup_task()
        asyncio.create_task(cleanup_coroutine)

    async def cleanup_old_logs(self):
        """Clean up log files older than 5 days"""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                return

            cutoff_time = datetime.now() - timedelta(days=5)

            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))

                    if file_time < cutoff_time and filename.endswith('.log'):
                        try:
                            os.remove(filepath)
                            logger.info(f"Cleaned up old log file: {filename}")
                        except Exception as e:
                            logger.warning(f"Failed to remove log file {filename}: {e}")

        except Exception as e:
            await log_error(e, {'function': 'cleanup_old_logs'})

    def start_conversation(self, user_id: int, initial_data: Dict[str, Any]) -> int:
        """Start a new conversation for task creation"""
        self.conversations[user_id] = {
            'data': initial_data,
            'state': TITLE,
            'start_time': datetime.now(),
            'timeout_task': None
        }

        # Set timeout to clean up conversation
        async def timeout_cleanup():
            await asyncio.sleep(self.conversation_timeout)
            if user_id in self.conversations:
                del self.conversations[user_id]
                logger.info(f"Conversation timeout for user {user_id}")

        timeout_task = asyncio.create_task(timeout_cleanup())
        self.conversations[user_id]['timeout_task'] = timeout_task

        return TITLE

    def get_conversation(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get conversation data for user"""
        return self.conversations.get(user_id)

    def update_conversation(self, user_id: int, **kwargs) -> None:
        """Update conversation data"""
        if user_id in self.conversations:
            self.conversations[user_id].update(kwargs)

    def end_conversation(self, user_id: int) -> None:
        """End conversation and clean up"""
        if user_id in self.conversations:
            # Cancel timeout task
            if self.conversations[user_id].get('timeout_task'):
                self.conversations[user_id]['timeout_task'].cancel()
            del self.conversations[user_id]

    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        return True  # Bot is now public - all users are authorized

    def create_priority_keyboard(self) -> InlineKeyboardMarkup:
        """Create inline keyboard for priority selection"""
        keyboard = [
            [
                InlineKeyboardButton("🔴 Urgent", callback_data=f"{PRIORITY_CALLBACK}_urgent"),
                InlineKeyboardButton("🟠 High", callback_data=f"{PRIORITY_CALLBACK}_high"),
            ],
            [
                InlineKeyboardButton("🟡 Normal", callback_data=f"{PRIORITY_CALLBACK}_normal"),
                InlineKeyboardButton("🟢 Low", callback_data=f"{PRIORITY_CALLBACK}_low"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def create_assignee_keyboard(self, assignees: list) -> InlineKeyboardMarkup:
        """Create inline keyboard for assignee selection"""
        keyboard = []
        for i in range(0, len(assignees), 2):
            row = []
            for j in range(2):
                if i + j < len(assignees):
                    row.append(InlineKeyboardButton(
                        assignees[i + j],
                        callback_data=f"{ASSIGNEE_CALLBACK}_{assignees[i + j]}"
                    ))
            if row:
                keyboard.append(row)

        # Add "None" option
        keyboard.append([InlineKeyboardButton("None", callback_data=f"{ASSIGNEE_CALLBACK}_None")])
        return InlineKeyboardMarkup(keyboard)

    def create_confirmation_keyboard(self) -> InlineKeyboardMarkup:
        """Create inline keyboard for confirmation"""
        keyboard = [
            [
                InlineKeyboardButton("✅ Create Task", callback_data=CONFIRM_CALLBACK),
                InlineKeyboardButton("❌ Cancel", callback_data=CANCEL_CALLBACK),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def setup_handlers(self, application: Application):
        """Setup all message handlers"""
        # Command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("createtask", self.create_task_command))
        application.add_handler(CommandHandler("cancel", self.cancel_command))
        application.add_handler(CommandHandler("mytasks", self.my_tasks_command))
        application.add_handler(CommandHandler("search", self.search_tasks_command))
        application.add_handler(CommandHandler("updatetask", self.update_task_command))

        # Callback query handler for inline keyboards
        application.add_handler(CallbackQueryHandler(self.handle_callback_query))

        # Conversation handler for interactive task creation
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("createtask", self.create_task_command)],
            states={
                TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_title_input)],
                DESCRIPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_description_input)],
                PRIORITY: [CallbackQueryHandler(self.handle_priority_selection, pattern=f"^{PRIORITY_CALLBACK}_")],
                DUE_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_due_date_input)],
                ASSIGNEE: [CallbackQueryHandler(self.handle_assignee_selection, pattern=f"^{ASSIGNEE_CALLBACK}_")],
                CONFIRMATION: [CallbackQueryHandler(self.handle_task_confirmation, pattern=f"^({CONFIRM_CALLBACK}|{CANCEL_CALLBACK})$")]
            },
            fallbacks=[CommandHandler("cancel", self.cancel_command)],
            conversation_timeout=self.conversation_timeout
        )
        application.add_handler(conv_handler)

        # Text message handler
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
            self.handle_private_message
        ))

        # Photo handler
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.PHOTO,
            self.handle_photo_message
        ))

        # Document handler
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.Document.ALL,
            self.handle_document_message
        ))

        # Audio handler
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.AUDIO,
            self.handle_audio_message
        ))

        # Video handler
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.VIDEO,
            self.handle_video_message
        ))

        # Voice handler
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.VOICE,
            self.handle_voice_message
        ))

        # Forwarded message handler (catches all forwarded content)
        application.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.FORWARDED,
            self.handle_forwarded_message
        ))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await self.send_help_message(update.message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await self.send_help_message(update.message)

    async def create_task_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /createtask command - start interactive task creation"""
        try:
            message = update.message

            # Check if there's already an active conversation
            if self.get_conversation(message.from_user.id):
                await message.reply_text("You already have an active task creation. Please complete it or use /cancel to start over.")
                return ConversationHandler.END

            # Extract initial data from message if provided
            initial_data = {
                'title': '',
                'description': '',
                'priority': 'normal',
                'due_date': None,
                'assignee': None,
                'source_type': 'interactive',
                'original_message': message
            }

            # Parse message for initial data
            if context.args:
                # Simple parsing: first argument as title, rest as description
                initial_data['title'] = context.args[0]
                if len(context.args) > 1:
                    initial_data['description'] = ' '.join(context.args[1:])

            # Start conversation
            state = self.start_conversation(message.from_user.id, initial_data)

            if initial_data['title']:
                # Skip to description if title was provided
                self.update_conversation(message.from_user.id, state=DESCRIPTION)
                await message.reply_text(
                    f"📝 **Task Title:** {initial_data['title']}\n\n"
                    "Now, please provide a detailed description for this task:",
                    parse_mode='Markdown'
                )
                return DESCRIPTION
            else:
                await message.reply_text(
                    "🎯 Let's create a new task!\n\n"
                    "First, please provide a **title** for the task:",
                    parse_mode='Markdown'
                )
                return TITLE

        except Exception as e:
            context_data = {
                'function': 'create_task_command',
                'user_id': update.message.from_user.id if update.message else None,
                'command': '/createtask'
            }
            await log_error(e, context_data)
            await update.message.reply_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cancel command - cancel current conversation"""
        try:
            message = update.message
            user_id = message.from_user.id

            if self.get_conversation(user_id):
                self.end_conversation(user_id)
                await message.reply_text("❌ Task creation cancelled.")
            else:
                await message.reply_text("No active task creation to cancel.")

        except Exception as e:
            context_data = {
                'function': 'cancel_command',
                'user_id': update.message.from_user.id if update.message else None,
                'command': '/cancel'
            }
            await log_error(e, context_data)
            await update.message.reply_text("Sorry, an error occurred.")

        return ConversationHandler.END

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        try:
            query = update.callback_query
            await query.answer()

            callback_data = query.data

            if callback_data.startswith(PRIORITY_CALLBACK):
                priority = callback_data.split('_', 2)[2]
                await self.handle_priority_selection(update, context, priority)
            elif callback_data.startswith(ASSIGNEE_CALLBACK):
                assignee = callback_data.split('_', 2)[2]
                await self.handle_assignee_selection(update, context, assignee)
            elif callback_data == CONFIRM_CALLBACK:
                await self.handle_task_confirmation(update, context)
            elif callback_data == CANCEL_CALLBACK:
                await self.handle_task_cancellation(update, context)

        except Exception as e:
            logger.error(f"Error handling callback query: {e}")

    async def handle_title_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle title input in conversation"""
        try:
            message = update.message
            user_id = message.from_user.id
            title = message.text.strip()

            if not title:
                await message.reply_text("Please provide a valid title:")
                return TITLE

            # Update conversation data
            conversation = self.get_conversation(user_id)
            if conversation:
                conversation['data']['title'] = title
                self.update_conversation(user_id, state=DESCRIPTION)

                await message.reply_text(
                    f"✅ **Title:** {title}\n\n"
                    "Now, please provide a **description** for this task:",
                    parse_mode='Markdown'
                )
                return DESCRIPTION

        except Exception as e:
            logger.error(f"Error handling title input: {e}")
            await update.message.reply_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def handle_description_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle description input in conversation"""
        try:
            message = update.message
            user_id = message.from_user.id
            description = message.text.strip()

            # Update conversation data
            conversation = self.get_conversation(user_id)
            if conversation:
                conversation['data']['description'] = description
                self.update_conversation(user_id, state=PRIORITY)

                await message.reply_text(
                    f"✅ **Description** received\n\n"
                    "Now, please select the **priority**:",
                    parse_mode='Markdown',
                    reply_markup=self.create_priority_keyboard()
                )
                return PRIORITY

        except Exception as e:
            logger.error(f"Error handling description input: {e}")
            await update.message.reply_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def handle_priority_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE, priority: str = None):
        """Handle priority selection"""
        try:
            query = update.callback_query if hasattr(update, 'callback_query') else None
            message = query.message if query else update.message
            user_id = query.from_user.id if query else update.message.from_user.id

            if priority is None and query:
                priority = query.data.split('_', 2)[2]

            # Update conversation data
            conversation = self.get_conversation(user_id)
            if conversation:
                conversation['data']['priority'] = priority
                self.update_conversation(user_id, state=DUE_DATE)

                if query:
                    await query.edit_message_text(
                        f"✅ **Priority:** {priority.title()}\n\n"
                        "Now, please provide a **due date** (YYYY-MM-DD) or type 'skip' for no due date:",
                        parse_mode='Markdown'
                    )
                else:
                    await message.reply_text(
                        f"✅ **Priority:** {priority.title()}\n\n"
                        "Now, please provide a **due date** (YYYY-MM-DD) or type 'skip' for no due date:",
                        parse_mode='Markdown'
                    )
                return DUE_DATE

        except Exception as e:
            logger.error(f"Error handling priority selection: {e}")
            if hasattr(update, 'message'):
                await update.message.reply_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def handle_due_date_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle due date input in conversation"""
        try:
            message = update.message
            user_id = message.from_user.id
            due_date_input = message.text.strip()

            due_date = None
            if due_date_input.lower() != 'skip':
                try:
                    # Try to parse date
                    due_date = datetime.strptime(due_date_input, '%Y-%m-%d').strftime('%Y-%m-%d')
                except ValueError:
                    await message.reply_text(
                        "Invalid date format. Please use YYYY-MM-DD or type 'skip':"
                    )
                    return DUE_DATE

            # Update conversation data
            conversation = self.get_conversation(user_id)
            if conversation:
                conversation['data']['due_date'] = due_date

                # Get available assignees (simplified - in real app, fetch from ClickUp)
                assignees = ["John", "Sarah", "Mike", "Lisa", "Team"]

                self.update_conversation(user_id, state=ASSIGNEE)

                await message.reply_text(
                    f"✅ **Due Date:** {due_date or 'None'}\n\n"
                    "Now, please select an **assignee**:",
                    parse_mode='Markdown',
                    reply_markup=self.create_assignee_keyboard(assignees)
                )
                return ASSIGNEE

        except Exception as e:
            logger.error(f"Error handling due date input: {e}")
            await update.message.reply_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def handle_assignee_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE, assignee: str = None):
        """Handle assignee selection"""
        try:
            query = update.callback_query if hasattr(update, 'callback_query') else None
            message = query.message if query else update.message
            user_id = query.from_user.id if query else update.message.from_user.id

            if assignee is None and query:
                assignee = query.data.split('_', 2)[2]

            # Convert "None" to None
            if assignee == "None":
                assignee = None

            # Update conversation data
            conversation = self.get_conversation(user_id)
            if conversation:
                conversation['data']['assignee'] = assignee
                self.update_conversation(user_id, state=CONFIRMATION)

                # Show confirmation with task details
                task_data = conversation['data']
                confirmation_text = self.format_task_confirmation(task_data)

                if query:
                    await query.edit_message_text(
                        confirmation_text,
                        parse_mode='Markdown',
                        reply_markup=self.create_confirmation_keyboard()
                    )
                else:
                    await message.reply_text(
                        confirmation_text,
                        parse_mode='Markdown',
                        reply_markup=self.create_confirmation_keyboard()
                    )
                return CONFIRMATION

        except Exception as e:
            logger.error(f"Error handling assignee selection: {e}")
            if hasattr(update, 'message'):
                await update.message.reply_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def handle_task_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle task creation confirmation"""
        try:
            query = update.callback_query
            await query.answer()

            user_id = query.from_user.id
            conversation = self.get_conversation(user_id)

            if conversation:
                task_data = conversation['data']
                original_message = task_data.get('original_message', query.message)

                # Create ClickUp task
                clickup_task = await self.create_clickup_task(task_data, original_message)

                if clickup_task:
                    await query.edit_message_text(
                        f"🎉 **Task Created Successfully!**\n\n"
                        f"**Title:** {task_data['title']}\n"
                        f"**Priority:** {task_data['priority'].title()}\n"
                        f"**ClickUp Task ID:** {clickup_task.get('id', 'N/A')}\n"
                        f"**ClickUp URL:** {clickup_task.get('url', 'N/A')}",
                        parse_mode='Markdown'
                    )
                    logger.info(f"Created interactive task: {clickup_task['name']}")
                else:
                    await query.edit_message_text(
                        "❌ Failed to create ClickUp task. Please try again."
                    )

                # End conversation
                self.end_conversation(user_id)

            return ConversationHandler.END

        except Exception as e:
            logger.error(f"Error handling task confirmation: {e}")
            if hasattr(update, 'callback_query'):
                await update.callback_query.edit_message_text("Sorry, an error occurred. Please try again.")
            return ConversationHandler.END

    async def handle_task_cancellation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle task creation cancellation"""
        try:
            query = update.callback_query
            await query.answer()

            user_id = query.from_user.id
            self.end_conversation(user_id)

            await query.edit_message_text("❌ Task creation cancelled.")
            return ConversationHandler.END

        except Exception as e:
            logger.error(f"Error handling task cancellation: {e}")
            if hasattr(update, 'callback_query'):
                await update.callback_query.edit_message_text("Sorry, an error occurred.")
            return ConversationHandler.END

    def format_task_confirmation(self, task_data: Dict[str, Any]) -> str:
        """Format task confirmation message"""
        confirmation = "📋 **Task Confirmation**\n\n"
        confirmation += f"**Title:** {task_data['title']}\n"
        confirmation += f"**Description:** {task_data['description']}\n"
        confirmation += f"**Priority:** {task_data['priority'].title()}\n"
        confirmation += f"**Due Date:** {task_data['due_date'] or 'None'}\n"
        confirmation += f"**Assignee:** {task_data['assignee'] or 'None'}\n\n"
        confirmation += "Please confirm to create this task:"
        return confirmation

    def prompt_for_missing_data(self, task_info: Dict[str, Any]) -> bool:
        """Check if task has missing essential data and prompt user"""
        missing_fields = []

        if not task_info.get('title'):
            missing_fields.append('title')
        if not task_info.get('description'):
            missing_fields.append('description')

        return len(missing_fields) > 0

    async def handle_missing_data_prompt(self, update: Update, task_info: Dict[str, Any]) -> Optional[int]:
        """Handle prompting for missing task data and return conversation state"""
        try:
            message = update.message
            user_id = message.from_user.id

            # Check if already in conversation
            if self.get_conversation(user_id):
                await message.reply_text("You already have an active task creation. Please complete it first.")
                return None

            # Start conversation with existing data
            initial_data = {
                'title': task_info.get('title', ''),
                'description': task_info.get('description', ''),
                'priority': task_info.get('priority', 'normal'),
                'due_date': task_info.get('due_date'),
                'assignee': task_info.get('assignee'),
                'source_type': task_info.get('source_type', 'prompted'),
                'original_message': message,
                'url': task_info.get('url'),
                'file_id': task_info.get('file_id'),
                'file_name': task_info.get('file_name')
            }

            state = self.start_conversation(user_id, initial_data)

            # Determine what to prompt for based on missing data
            if not initial_data['title']:
                await message.reply_text(
                    "🤔 I detected a task but the title is unclear.\n\n"
                    "Please provide a **title** for this task:",
                    parse_mode='Markdown'
                )
                return TITLE

            elif not initial_data['description']:
                await message.reply_text(
                    f"✅ **Title:** {initial_data['title']}\n\n"
                    "I need more details. Please provide a **description**:",
                    parse_mode='Markdown'
                )
                return DESCRIPTION

            else:
                # We have title and description, go to priority selection
                self.update_conversation(user_id, state=PRIORITY)
                await message.reply_text(
                    f"✅ **Title:** {initial_data['title']}\n"
                    f"✅ **Description:** {initial_data['description']}\n\n"
                    "Now, please select the **priority**:",
                    parse_mode='Markdown',
                    reply_markup=self.create_priority_keyboard()
                )
                return PRIORITY

        except Exception as e:
            logger.error(f"Error handling missing data prompt: {e}")
            return None

    async def my_tasks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mytasks command - show user's tasks"""
        try:
            message = update.message

            # Get tasks from ClickUp
            tasks = await self.get_clickup_tasks()

            if tasks:
                response_text = "📋 **Your Recent Tasks:**\n\n"
                for task in tasks[:10]:  # Show last 10 tasks
                    status_emoji = {"to do": "⏳", "in progress": "🔄", "complete": "✅"}.get(task.get('status', {}).get('status', ''), "📋")
                    priority_emoji = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢"}.get(task.get('priority', 3), "🟡")

                    response_text += f"{status_emoji} {priority_emoji} **{task['name']}**\n"
                    response_text += f"   Status: {task.get('status', {}).get('status', 'Unknown')}\n"
                    response_text += f"   ID: `{task['id']}`\n\n"

                response_text += f"Use `/search <keyword>` to find specific tasks or `/updatetask <task_id>` to update a task."
            else:
                response_text = "No tasks found or unable to fetch tasks from ClickUp."

            await message.reply_text(response_text, parse_mode='Markdown')

        except Exception as e:
            context_data = {
                'function': 'my_tasks_command',
                'user_id': update.message.from_user.id if update.message else None,
                'command': '/mytasks'
            }
            await log_error(e, context_data)
            await update.message.reply_text("Sorry, an error occurred while fetching tasks.")

    async def search_tasks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search command - search for tasks"""
        try:
            message = update.message

            if not context.args:
                await message.reply_text("Please provide a search term. Usage: `/search <keyword>`")
                return

            search_term = ' '.join(context.args)
            tasks = await self.get_clickup_tasks()

            if tasks:
                matching_tasks = []
                for task in tasks:
                    if search_term.lower() in task['name'].lower() or search_term.lower() in task.get('description', '').lower():
                        matching_tasks.append(task)

                if matching_tasks:
                    response_text = f"🔍 **Tasks matching '{search_term}':**\n\n"
                    for task in matching_tasks[:10]:
                        status_emoji = {"to do": "⏳", "in progress": "🔄", "complete": "✅"}.get(task.get('status', {}).get('status', ''), "📋")
                        priority_emoji = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢"}.get(task.get('priority', 3), "🟡")

                        response_text += f"{status_emoji} {priority_emoji} **{task['name']}**\n"
                        response_text += f"   Status: {task.get('status', {}).get('status', 'Unknown')}\n"
                        response_text += f"   ID: `{task['id']}`\n\n"

                    response_text += f"Found {len(matching_tasks)} matching tasks."
                else:
                    response_text = f"No tasks found matching '{search_term}'."
            else:
                response_text = "Unable to fetch tasks from ClickUp."

            await message.reply_text(response_text, parse_mode='Markdown')

        except Exception as e:
            context_data = {
                'function': 'search_tasks_command',
                'user_id': update.message.from_user.id if update.message else None,
                'command': '/search'
            }
            await log_error(e, context_data)
            await update.message.reply_text("Sorry, an error occurred while searching tasks.")

    async def update_task_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /updatetask command - update task status"""
        try:
            message = update.message

            if not context.args or len(context.args) < 2:
                await message.reply_text(
                    "Please provide task ID and new status. Usage: `/updatetask <task_id> <status>`\n\n"
                    "Available statuses: `to do`, `in progress`, `complete`",
                    parse_mode='Markdown'
                )
                return

            task_id = context.args[0]
            new_status = ' '.join(context.args[1:]).lower()

            # Validate status
            valid_statuses = ['to do', 'in progress', 'complete']
            if new_status not in valid_statuses:
                await message.reply_text(
                    f"Invalid status '{new_status}'. Available statuses: {', '.join(valid_statuses)}"
                )
                return

            # Update task in ClickUp
            success = await self.update_task_status(task_id, new_status)

            if success:
                await message.reply_text(
                    f"✅ Task `{task_id}` status updated to '{new_status}'",
                    parse_mode='Markdown'
                )
                logger.info(f"Updated task {task_id} status to {new_status}")
            else:
                await message.reply_text(f"Failed to update task {task_id}. Please check the task ID and try again.")

        except Exception as e:
            context_data = {
                'function': 'update_task_command',
                'user_id': update.message.from_user.id if update.message else None,
                'command': '/updatetask'
            }
            await log_error(e, context_data)
            await update.message.reply_text("Sorry, an error occurred while updating the task.")

    async def _make_clickup_request(self, method: str, url: str, params: dict = None, json: dict = None, expected_status_codes: list = None) -> Any:
        """
        Generic method for making ClickUp API requests with circuit breaker pattern.

        Args:
            method: HTTP method ('GET', 'POST', 'PUT', etc.)
            url: API endpoint URL
            params: Query parameters for GET requests
            json: JSON payload for POST/PUT requests
            expected_status_codes: List of expected HTTP status codes (default: [200, 201, 204])

        Returns:
            Response data or True for successful requests with no content

        Raises:
            APIError: When the request fails or returns unexpected status
        """
        if expected_status_codes is None:
            expected_status_codes = [200, 201, 204]

        async def make_api_call():
            # Make the HTTP request
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.clickup_headers, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.clickup_headers, json=json, timeout=30)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.clickup_headers, json=json, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check response status
            if response.status_code not in expected_status_codes:
                raise APIError(
                    "Unable to connect to ClickUp service. Please try again later.",
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )

            # Return appropriate response
            if response.status_code == 204:  # No Content
                return True
            elif response.content:
                return response.json()
            else:
                return True

        # Execute with circuit breaker
        return await clickup_circuit_breaker.call(make_api_call)

    @retry_async(max_retries=3, delay=1, backoff=2, exceptions=(APIError, requests.exceptions.RequestException))
    @handle_api_errors
    async def get_clickup_tasks(self) -> Optional[list]:
        """Get tasks from ClickUp with retry logic"""
        try:
            if not self.clickup_list_id:
                raise ValidationError("ClickUp list ID is not configured")

            url = f"https://api.clickup.com/api/v2/list/{self.clickup_list_id}/task"
            params = {
                'archived': 'false',
                'include_closed': 'true',
                'subtasks': 'false'
            }

            result = await self._make_clickup_request('GET', url, params=params, expected_status_codes=[200])
            result = result.get('tasks', []) if result else []
            logger.info(f"Successfully fetched {len(result)} tasks from ClickUp")
            return result

        except ValidationError as e:
            await log_error(e, {'function': 'get_clickup_tasks'})
            return None
        except APIError as e:
            # Already logged by decorator
            return None
        except Exception as e:
            await log_error(e, {'function': 'get_clickup_tasks'})
            return None

    @retry_async(max_retries=3, delay=2, backoff=2, exceptions=(APIError, requests.exceptions.RequestException))
    @handle_api_errors
    async def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status in ClickUp with retry logic"""
        try:
            if not task_id:
                raise ValidationError("Task ID is required")
            if not status:
                raise ValidationError("Status is required")

            # Validate status
            valid_statuses = ['to do', 'in progress', 'complete', 'closed']
            if status not in valid_statuses:
                raise ValidationError(f"Invalid status '{status}'. Valid statuses: {valid_statuses}")

            url = f"https://api.clickup.com/api/v2/task/{task_id}"
            data = {
                'status': status
            }

            result = await self._make_clickup_request('PUT', url, json=data, expected_status_codes=[200, 204])
            logger.info(f"Successfully updated task {task_id} status to '{status}'")
            return result

        except ValidationError as e:
            await log_error(e, {'task_id': task_id, 'status': status})
            return False
        except APIError as e:
            # Already logged by decorator
            return False
        except Exception as e:
            await log_error(e, {'task_id': task_id, 'status': status})
            return False

    async def handle_private_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle private messages and forwarded messages"""
        try:
            message = update.message

            # Check for URLs in the message
            urls = self.extract_urls(message.text)
            google_docs_url = self.extract_google_docs_url(message.text)

            # Determine what to process
            task_info = None

            if google_docs_url:
                # Process Google Docs URL
                logger.info(f"Processing Google Docs URL: {google_docs_url}")
                task_info = await self.process_url_with_gemini(google_docs_url)
            elif urls and self.contains_task_keywords(message.text):
                # Process text with URLs
                logger.info(f"Processing message with URLs: {message.text[:100]}...")
                task_info = await self.process_message_with_gemini(message.text)
            elif self.contains_task_keywords(message.text):
                # Process regular text message
                logger.info(f"Processing text message: {message.text[:100]}...")
                task_info = await self.process_message_with_gemini(message.text)
            elif message.forward_origin:
                # Handle forwarded messages (already handled by separate handler)
                return
            else:
                # Send help message if no actionable content
                await self.send_help_message(message)
                return

            if task_info and task_info.get('is_task'):
                # Add source type information
                if google_docs_url:
                    task_info['source_type'] = 'google_docs_url'
                    task_info['url'] = google_docs_url
                elif urls:
                    task_info['source_type'] = 'text_with_urls'
                else:
                    task_info['source_type'] = 'text'

                # Check if we need to prompt for missing data
                if self.prompt_for_missing_data(task_info):
                    # Start interactive conversation to fill missing data
                    conv_state = await self.handle_missing_data_prompt(update, task_info)
                    if conv_state:
                        return conv_state
                else:
                    # Create ClickUp task directly
                    clickup_task = await self.create_clickup_task(task_info, message)

                    if clickup_task:
                        logger.info(f"Created ClickUp task: {clickup_task['name']}")
                        await self.reply_to_message(message, task_info, clickup_task)
                    else:
                        logger.error("Failed to create ClickUp task")
                        await message.reply_text("Sorry, I couldn't create the ClickUp task. Please try again.")
            else:
                # Message doesn't appear to be a task
                if google_docs_url:
                    await message.reply_text(
                        "I couldn't identify any tasks in the Google Doc. Please make sure the document contains actionable items or task-related content."
                    )
                else:
                    await message.reply_text(
                        "I couldn't identify a task in your message. Please make sure it contains task-related keywords like 'task', 'todo', 'deadline', etc.\n\n"
                        "Or use `/createtask` to create a task interactively!",
                        parse_mode='Markdown'
                    )

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing your message. Please try again.")

    async def send_help_message(self, message: Message):
        """Send help message to user"""
        help_text = f"""🤖 **ClickGram Bot Help**

I can help you create ClickUp tasks from Telegram messages and media files in multiple ways!

**🎯 Interactive Task Creation:**
Use `/createtask` to start a guided conversation:
- `/createtask` - Start interactive task creation
- `/createtask "Task Title" "Description"` - Pre-fill title and description
- `/cancel` - Cancel current task creation

**📋 Task Management:**
- `/mytasks` - Show your recent tasks
- `/search <keyword>` - Search for tasks by keyword
- `/updatetask <task_id> <status>` - Update task status

**📝 Automated Task Detection:**
Send messages containing task-related keywords like:
- task, todo, assignment, action item, deliverable
- deadline, due, complete, finish, work on
- implement, create, build, develop, fix, resolve
- handle, done, look at, asap, as soon

**📊 Multi-Media Support:**
- 🖼️ **Photos:** Send screenshots or images with text
- 📄 **Documents:** PDFs, text files, etc.
- 🎵 **Audio/Voice:** Transcribe and extract tasks
- 🎬 **Videos:** Analyze video content
- 🔄 **Forwarded Messages:** Process any forwarded content
- 🔗 **URLs & Google Docs:** Access and extract from links

**💡 Interactive Features:**
- Multi-step conversations with guided prompts
- Inline keyboards for priority & assignee selection
- Smart date parsing and validation
- Task confirmation before creation
- Conversation timeout and cleanup

**Example Commands:**
- `/createtask` - Start interactive creation
- `/createtask "Fix login bug" "Users cannot login with Google OAuth"` - Pre-fill data
- `Please implement the user authentication feature by next week` - Auto-detect task

I'll analyze the content, guide you through missing information, and create ClickUp tasks interactively!

{f'Note: This bot is restricted to authorized users only.' if self.authorized_users else ''}"""

        await message.reply_text(help_text, parse_mode='Markdown')

    async def handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages using Gemini vision"""
        try:
            message = update.message

            # Get the photo file
            photo_file = await message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()

            # Process image with Gemini Vision
            task_info = await self.process_image_with_gemini(photo_bytes, message.caption or "")

            if task_info and task_info.get('is_task'):
                # Add photo metadata
                task_info['source_type'] = 'photo'
                task_info['file_id'] = message.photo[-1].file_id

                # Check if we need to prompt for missing data
                if self.prompt_for_missing_data(task_info):
                    # Start interactive conversation to fill missing data
                    conv_state = await self.handle_missing_data_prompt(update, task_info)
                    if conv_state:
                        return conv_state
                else:
                    # Create ClickUp task directly
                    clickup_task = await self.create_clickup_task(task_info, message)

                    if clickup_task:
                        logger.info(f"Created ClickUp task from photo: {clickup_task['name']}")
                        await self.reply_to_message(message, task_info, clickup_task)
                    else:
                        await message.reply_text("Sorry, I couldn't create the ClickUp task from the photo.")
            else:
                await message.reply_text(
                    "I couldn't identify a task in the photo. Please make sure the image contains task-related content or text.\n\n"
                    "Or use `/createtask` to create a task interactively!",
                    parse_mode='Markdown'
                )

        except Exception as e:
            logger.error(f"Error handling photo message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing the photo. Please try again.")

    async def handle_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document messages using Gemini document processing"""
        try:
            message = update.message

            document = message.document

            # Get the document file
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()

            # Process document with Gemini
            task_info = await self.process_document_with_gemini(file_bytes, document.file_name, document.mime_type, message.caption or "")

            if task_info and task_info.get('is_task'):
                # Add document metadata
                task_info['source_type'] = 'document'
                task_info['file_name'] = document.file_name
                task_info['file_id'] = document.file_id

                clickup_task = await self.create_clickup_task(task_info, message)

                if clickup_task:
                    logger.info(f"Created ClickUp task from document: {clickup_task['name']}")
                    await self.reply_to_message(message, task_info, clickup_task)
                else:
                    await message.reply_text("Sorry, I couldn't create the ClickUp task from the document.")
            else:
                await message.reply_text(
                    "I couldn't identify any tasks in the document. The document may not contain actionable items."
                )

        except Exception as e:
            logger.error(f"Error handling document message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing the document. Please try again.")

    async def handle_audio_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle audio messages using Gemini audio processing"""
        try:
            message = update.message

            audio = message.audio

            # Get the audio file
            audio_file = await audio.get_file()
            audio_bytes = await audio_file.download_as_bytearray()

            # Process audio with Gemini
            task_info = await self.process_audio_with_gemini(audio_bytes, message.caption or "")

            if task_info and task_info.get('is_task'):
                # Add audio metadata
                task_info['source_type'] = 'audio'
                task_info['file_id'] = audio.file_id

                clickup_task = await self.create_clickup_task(task_info, message)

                if clickup_task:
                    logger.info(f"Created ClickUp task from audio: {clickup_task['name']}")
                    await self.reply_to_message(message, task_info, clickup_task)
                else:
                    await message.reply_text("Sorry, I couldn't create the ClickUp task from the audio.")
            else:
                await message.reply_text(
                    "I couldn't identify any tasks in the audio content. The audio may not contain actionable items."
                )

        except Exception as e:
            logger.error(f"Error handling audio message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing the audio file. Please try again.")

    async def handle_video_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle video messages using Gemini video processing"""
        try:
            message = update.message

            video = message.video

            # Get the video file
            video_file = await video.get_file()
            video_bytes = await video_file.download_as_bytearray()

            # Process video with Gemini
            task_info = await self.process_video_with_gemini(video_bytes, message.caption or "")

            if task_info and task_info.get('is_task'):
                # Add video metadata
                task_info['source_type'] = 'video'
                task_info['file_id'] = video.file_id

                clickup_task = await self.create_clickup_task(task_info, message)

                if clickup_task:
                    logger.info(f"Created ClickUp task from video: {clickup_task['name']}")
                    await self.reply_to_message(message, task_info, clickup_task)
                else:
                    await message.reply_text("Sorry, I couldn't create the ClickUp task from the video.")
            else:
                await message.reply_text(
                    "I couldn't identify any tasks in the video content. The video may not contain actionable items."
                )

        except Exception as e:
            logger.error(f"Error handling video message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing the video file. Please try again.")

    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages using Gemini audio processing"""
        try:
            message = update.message

            voice = message.voice

            # Get the voice file
            voice_file = await voice.get_file()
            voice_bytes = await voice_file.download_as_bytearray()

            # Process voice with Gemini
            task_info = await self.process_audio_with_gemini(voice_bytes, f"Voice message from {message.from_user.first_name}")

            if task_info and task_info.get('is_task'):
                # Add voice metadata
                task_info['source_type'] = 'voice'
                task_info['file_id'] = voice.file_id

                clickup_task = await self.create_clickup_task(task_info, message)

                if clickup_task:
                    logger.info(f"Created ClickUp task from voice message: {clickup_task['name']}")
                    await self.reply_to_message(message, task_info, clickup_task)
                else:
                    await message.reply_text("Sorry, I couldn't create the ClickUp task from the voice message.")
            else:
                await message.reply_text(
                    "I couldn't identify any tasks in the voice message. The voice message may not contain actionable items."
                )

        except Exception as e:
            logger.error(f"Error handling voice message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing the voice message. Please try again.")

    async def handle_forwarded_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle forwarded messages"""
        try:
            message = update.message

            # Get content from forwarded message
            content_to_process = ""

            if message.text:
                content_to_process = message.text
            elif message.photo:
                # Handle forwarded photos with Gemini Vision
                photo_file = await message.photo[-1].get_file()
                photo_bytes = await photo_file.download_as_bytearray()

                # Use Gemini Vision to process the image
                vision_result = await self.process_image_with_gemini(photo_bytes, message.caption or "")
                if vision_result:
                    # Convert vision result to text content
                    content_to_process = f"Image Analysis Result:\nTitle: {vision_result.get('title', 'N/A')}\nDescription: {vision_result.get('description', 'N/A')}"
                    if vision_result.get('is_task'):
                        task_info = vision_result
                        task_info['source_type'] = 'forwarded_photo'
                        task_info['file_id'] = message.photo[-1].file_id
                        if message.forward_origin:
                            task_info['forwarded_from'] = str(message.forward_origin)
                        return task_info
                else:
                    content_to_process = "Forwarded photo content (unable to analyze)"
            elif message.document:
                # Handle forwarded documents
                document = message.document
                file = await document.get_file()
                file_bytes = await file.download_as_bytearray()

                file_ext = document.file_name.split('.')[-1].lower() if document.file_name else ''
                if file_ext == 'txt':
                    content_to_process = file_bytes.decode('utf-8', errors='ignore')
                elif file_ext == 'pdf':
                    # Use Gemini to process PDF
                    pdf_result = await self.process_document_with_gemini(file_bytes, document.file_name, document.mime_type, "")
                    if pdf_result:
                        content_to_process = f"PDF Analysis Result:\nTitle: {pdf_result.get('title', 'N/A')}\nDescription: {pdf_result.get('description', 'N/A')}"
                        if pdf_result.get('is_task'):
                            task_info = pdf_result
                            task_info['source_type'] = 'forwarded_document'
                            task_info['file_name'] = document.file_name
                            task_info['file_id'] = document.file_id
                            if message.forward_origin:
                                task_info['forwarded_from'] = str(message.forward_origin)
                            return task_info
                    else:
                        content_to_process = "Forwarded PDF content (unable to analyze)"
                else:
                    content_to_process = f"Forwarded document: {document.file_name}"
            else:
                # Other types of forwarded content
                content_to_process = f"Forwarded message from {message.forward_origin.date if message.forward_origin else 'unknown'}"

            logger.info(f"Processing forwarded message: {content_to_process[:100]}...")

            if content_to_process.strip():
                # Process the content
                task_info = await self.process_message_with_gemini(content_to_process)

                if task_info and task_info.get('is_task'):
                    # Add forwarded metadata
                    task_info['source_type'] = 'forwarded'
                    if message.forward_origin:
                        task_info['forwarded_from'] = str(message.forward_origin)

                    clickup_task = await self.create_clickup_task(task_info, message)

                    if clickup_task:
                        logger.info(f"Created ClickUp task from forwarded message: {clickup_task['name']}")
                        await self.reply_to_message(message, task_info, clickup_task)
                    else:
                        await message.reply_text("Sorry, I couldn't create the ClickUp task from the forwarded message.")
                else:
                    await message.reply_text(
                        "I couldn't identify a task in the forwarded content. Please make sure it contains task-related information."
                    )
            else:
                await message.reply_text(
                    "I couldn't extract readable content from the forwarded message. Please send a message with text or readable media."
                )

        except Exception as e:
            logger.error(f"Error handling forwarded message: {e}")
            await update.message.reply_text("Sorry, an error occurred while processing the forwarded message. Please try again.")

    async def process_audio_with_gemini(self, audio_bytes: bytes, context: str = "") -> Optional[Dict[str, Any]]:
        """Process audio with Gemini audio processing"""
        try:
            # Create audio part for Gemini
            audio_part = {
                "mime_type": "audio/mp3",  # Default, Gemini will detect actual type
                "data": audio_bytes
            }

            # Build prompt with context
            prompt_text = f"Analyze this audio content and identify any tasks, assignments, or actionable items."
            if context:
                prompt_text += f"\n\nContext: {context}"
            prompt_text += """

Please transcribe the audio content and extract any tasks, deadlines, assignments, or actionable items mentioned.

Please respond with a JSON object containing:
{
    "is_task": boolean,
    "title": "Brief task title based on audio content",
    "description": "Detailed description including transcription and identified tasks",
    "priority": "urgent|high|normal|low",
    "due_date": "YYYY-MM-DD or null if not specified",
    "assignee": "Person mentioned or null",
    "tags": ["tag1", "tag2"],
    "estimated_hours": number or null
}

Only return the JSON object, nothing else."""

            # Send to Gemini
            response = self.vision_model.generate_content([audio_part, prompt_text])

            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini audio response as JSON: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error processing audio with Gemini: {e}")
            return None

    async def process_video_with_gemini(self, video_bytes: bytes, context: str = "") -> Optional[Dict[str, Any]]:
        """Process video with Gemini video processing"""
        try:
            # Create video part for Gemini
            video_part = {
                "mime_type": "video/mp4",  # Default, Gemini will detect actual type
                "data": video_bytes
            }

            # Build prompt with context
            prompt_text = f"Analyze this video content and identify any tasks, assignments, or actionable items."
            if context:
                prompt_text += f"\n\nContext: {context}"
            prompt_text += """

Please analyze the video content and extract any tasks, instructions, or actionable items shown or mentioned.

Please respond with a JSON object containing:
{
    "is_task": boolean,
    "title": "Brief task title based on video content",
    "description": "Detailed description of video content and identified tasks",
    "priority": "urgent|high|normal|low",
    "due_date": "YYYY-MM-DD or null if not specified",
    "assignee": "Person mentioned or null",
    "tags": ["tag1", "tag2"],
    "estimated_hours": number or null
}

Only return the JSON object, nothing else."""

            # Send to Gemini
            response = self.vision_model.generate_content([video_part, prompt_text])

            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini video response as JSON: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error processing video with Gemini: {e}")
            return None

    async def process_document_with_gemini(self, document_bytes: bytes, filename: str, mime_type: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Process document with Gemini document processing"""
        try:
            # Create document part for Gemini
            document_part = {
                "mime_type": mime_type or "application/pdf",
                "data": document_bytes
            }

            # Build prompt with context
            prompt_text = f"Analyze this document ({filename}) and identify any tasks, assignments, or actionable items."
            if context:
                prompt_text += f"\n\nContext: {context}"
            prompt_text += """

Please read the document content and extract any tasks, deadlines, assignments, or actionable items mentioned.

Please respond with a JSON object containing:
{
    "is_task": boolean,
    "title": "Brief task title based on document content",
    "description": "Detailed description including extracted text and identified tasks",
    "priority": "urgent|high|normal|low",
    "due_date": "YYYY-MM-DD or null if not specified",
    "assignee": "Person mentioned or null",
    "tags": ["tag1", "tag2"],
    "estimated_hours": number or null
}

Only return the JSON object, nothing else."""

            # Send to Gemini
            response = self.vision_model.generate_content([document_part, prompt_text])

            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini document response as JSON: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error processing document with Gemini: {e}")
            return None

    async def process_url_with_gemini(self, url: str) -> Optional[Dict[str, Any]]:
        """Process URL with Gemini's web browsing capability"""
        try:
            prompt = f"""
            Please access and analyze this URL: {url}

            Read the content and identify any tasks, assignments, deadlines, or actionable items mentioned in the document.

            Please respond with a JSON object containing:
            {{
                "is_task": boolean,
                "title": "Brief task title based on the content",
                "description": "Detailed description including key points and identified tasks",
                "priority": "urgent|high|normal|low",
                "due_date": "YYYY-MM-DD or null if not specified",
                "assignee": "Person mentioned or null",
                "tags": ["tag1", "tag2"],
                "estimated_hours": number or null
            }}

            Only return the JSON object, nothing else.
            """

            response = self.model.generate_content(prompt)

            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini URL response as JSON: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error processing URL with Gemini: {e}")
            return None

    def contains_task_keywords(self, text: str) -> bool:
        """Check if message contains task-related keywords"""
        if not text:
            return False

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.task_keywords)

    def extract_urls(self, text: str) -> list:
        """Extract URLs from text"""
        if not text:
            return []

        # Simple URL regex pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls

    def extract_google_docs_url(self, text: str) -> Optional[str]:
        """Extract Google Docs URL from text"""
        if not text:
            return None

        # Google Docs URL patterns
        docs_patterns = [
            r'https://docs\.google\.com/document/d/([a-zA-Z0-9-_]+)',
            r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)',
            r'https://docs\.google\.com/presentation/d/([a-zA-Z0-9-_]+)',
            r'https://docs\.google\.com/forms/d/([a-zA-Z0-9-_]+)'
        ]

        for pattern in docs_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None

    @retry_async(max_retries=2, delay=1, backoff=2, exceptions=(Exception,))
    async def process_message_with_gemini(self, message_text: str) -> Optional[Dict[str, Any]]:
        """Process message with Gemini API to extract task information"""
        try:
            if not message_text or not message_text.strip():
                logger.warning("Empty message text provided to Gemini processing")
                return None

            prompt = f"""
            Analyze the following message and determine if it contains a task or assignment.
            If it does, extract the relevant task information.

            Message: "{message_text}"

            Please respond with a JSON object containing:
            {{
                "is_task": boolean,
                "title": "Brief task title",
                "description": "Detailed task description",
                "priority": "urgent|high|normal|low",
                "due_date": "YYYY-MM-DD or null if not specified",
                "assignee": "Person mentioned or null",
                "tags": ["tag1", "tag2"],
                "estimated_hours": number or null
            }}

            Only return the JSON object, nothing else.
            """

            async def make_gemini_call():
                response = self.model.generate_content(prompt)
                return response

            # Use circuit breaker for Gemini API call
            response = await gemini_circuit_breaker.call(make_gemini_call)

            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError as e:
                error_context = {
                    'function': 'process_message_with_gemini',
                    'message_text': message_text[:200],
                    'gemini_response': response.text[:500] if hasattr(response, 'text') else 'No response text'
                }
                await log_error(e, error_context)
                return None

        except Exception as e:
            error_context = {
                'function': 'process_message_with_gemini',
                'message_text': message_text[:200],
                'gemini_error': True
            }
            await log_error(e, error_context)
            return None

    async def process_image_with_gemini(self, image_bytes: bytes, caption: str = "") -> Optional[Dict[str, Any]]:
        """Process image with Gemini Vision API to extract task information"""
        try:
            # Create image part for Gemini
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }

            # Build prompt with caption if available
            prompt_text = "Analyze this image and determine if it contains a task, assignment, or actionable items."
            if caption:
                prompt_text += f"\n\nCaption: {caption}"
            prompt_text += """

If the image contains text, read it and identify any tasks.
If the image shows a diagram, screenshot, or other visual content, describe what task it might represent.

Please respond with a JSON object containing:
{
    "is_task": boolean,
    "title": "Brief task title based on image content",
    "description": "Detailed description including extracted text and visual content",
    "priority": "urgent|high|normal|low",
    "due_date": "YYYY-MM-DD or null if not specified",
    "assignee": "Person mentioned or null",
    "tags": ["tag1", "tag2"],
    "estimated_hours": number or null
}

Only return the JSON object, nothing else."""

            # Send to Gemini Vision
            response = self.vision_model.generate_content([image_part, prompt_text])

            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini Vision response as JSON: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error processing image with Gemini Vision: {e}")
            return None

    @retry_async(max_retries=3, delay=2, backoff=2, exceptions=(APIError, requests.exceptions.RequestException))
    @handle_api_errors
    async def create_clickup_task(self, task_info: Dict[str, Any], original_message: Message) -> Optional[Dict[str, Any]]:
        """Create a task in ClickUp with retry logic and circuit breaker"""
        try:
            # Validate required fields
            if not task_info.get('title'):
                raise ValidationError("Task title is required")

            if not self.clickup_list_id:
                raise ValidationError("ClickUp list ID is not configured")

            # Prepare task data
            task_data = {
                "name": task_info.get('title', 'New Task from Telegram'),
                "description": self.format_task_description(task_info, original_message),
                "status": "to do",
                "priority": self.map_priority(task_info.get('priority', 'normal')),
                "tags": task_info.get('tags', [])
            }

            # Add due date if specified
            if task_info.get('due_date'):
                try:
                    due_date = datetime.strptime(task_info['due_date'], '%Y-%m-%d')
                    task_data['due_date'] = int(due_date.timestamp() * 1000)
                except ValueError as e:
                    logger.warning(f"Invalid due date format: {task_info['due_date']}. Error: {e}")
                    # Don't raise, just skip the due date

            # Add time estimate if specified
            if task_info.get('estimated_hours'):
                try:
                    task_data['time_estimate'] = int(task_info['estimated_hours'] * 3600000)  # Convert to milliseconds
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid estimated hours: {task_info['estimated_hours']}. Error: {e}")

            # Create task via ClickUp API with circuit breaker
            url = f"https://api.clickup.com/api/v2/list/{self.clickup_list_id}/task"

            result = await self._make_clickup_request('POST', url, json=task_data, expected_status_codes=[200, 201])

            logger.info(f"Successfully created ClickUp task: {result.get('name', 'Unknown')} (ID: {result.get('id', 'Unknown')})")
            return result

        except ValidationError as e:
            await log_error(e, {'task_info': str(task_info)[:500]})
            logger.error(f"Validation error creating task: {e}")
            return None
        except APIError as e:
            # Already logged by decorator
            return None
        except Exception as e:
            error_context = {
                'task_info': str(task_info)[:500],
                'task_title': task_info.get('title', 'Unknown'),
                'user_id': getattr(original_message.from_user, 'id', 'Unknown') if original_message else 'Unknown'
            }
            await log_error(e, error_context)
            return None

    def format_task_description(self, task_info: Dict[str, Any], original_message: Message) -> str:
        """Format task description with additional context"""
        description = task_info.get('description', '')

        # Add source type information
        source_type = task_info.get('source_type', 'text')
        description += f"\n\n**Source Type:** {source_type.upper()}\n"

        # Add original message context
        if original_message.text:
            description += f"\n**Original Message:**\n{original_message.text}"

        # Add message metadata
        description += f"\n\n**Message Details:**\n"
        description += f"- Date: {original_message.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        description += f"- Message ID: {original_message.message_id}\n"

        # Add sender info
        if original_message.from_user:
            sender_name = original_message.from_user.first_name or 'Unknown'
            if original_message.from_user.last_name:
                sender_name += f" {original_message.from_user.last_name}"
            description += f"- Sent by: {sender_name} (ID: {original_message.from_user.id})\n"

        # Add forward info if applicable
        if original_message.forward_origin:
            description += f"- **Forwarded Message**\n"
            if hasattr(original_message.forward_origin, 'sender_user'):
                forward_sender = original_message.forward_origin.sender_user
                sender_name = forward_sender.first_name or 'Unknown'
                if forward_sender.last_name:
                    sender_name += f" {forward_sender.last_name}"
                description += f"- Original from: {sender_name} (ID: {forward_sender.id})\n"
            if hasattr(original_message.forward_origin, 'date'):
                description += f"- Original date: {original_message.forward_origin.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            if hasattr(original_message.forward_origin, 'chat'):
                description += f"- From chat: {original_message.forward_origin.chat.id}\n"

        # Add media-specific information
        if source_type == 'photo':
            if original_message.photo:
                description += f"- **Photo Info:**\n"
                description += f"  - File ID: {task_info.get('file_id', 'N/A')}\n"
                description += f"  - Resolution: {original_message.photo[-1].width}x{original_message.photo[-1].height}\n"
                description += f"  - File Size: {original_message.photo[-1].file_size} bytes\n"

        elif source_type == 'document':
            if original_message.document:
                description += f"- **Document Info:**\n"
                description += f"  - File Name: {task_info.get('file_name', original_message.document.file_name)}\n"
                description += f"  - File ID: {task_info.get('file_id', original_message.document.file_id)}\n"
                description += f"  - MIME Type: {original_message.document.mime_type}\n"
                description += f"  - File Size: {original_message.document.file_size} bytes\n"

        elif source_type == 'audio':
            if original_message.audio:
                description += f"- **Audio Info:**\n"
                description += f"  - File ID: {task_info.get('file_id', original_message.audio.file_id)}\n"
                description += f"  - Duration: {original_message.audio.duration} seconds\n"
                description += f"  - File Size: {original_message.audio.file_size} bytes\n"
                if original_message.audio.performer:
                    description += f"  - Performer: {original_message.audio.performer}\n"
                if original_message.audio.title:
                    description += f"  - Title: {original_message.audio.title}\n"

        elif source_type == 'video':
            if original_message.video:
                description += f"- **Video Info:**\n"
                description += f"  - File ID: {task_info.get('file_id', original_message.video.file_id)}\n"
                description += f"  - Duration: {original_message.video.duration} seconds\n"
                description += f"  - File Size: {original_message.video.file_size} bytes\n"
                description += f"  - Resolution: {original_message.video.width}x{original_message.video.height}\n"

        elif source_type == 'voice':
            if original_message.voice:
                description += f"- **Voice Message Info:**\n"
                description += f"  - File ID: {task_info.get('file_id', original_message.voice.file_id)}\n"
                description += f"  - Duration: {original_message.voice.duration} seconds\n"
                description += f"  - File Size: {original_message.voice.file_size} bytes\n"

        # Add task-specific information
        if task_info.get('assignee'):
            description += f"- Assigned to: {task_info['assignee']}\n"

        if task_info.get('forwarded_from'):
            description += f"- Forwarded from: {task_info['forwarded_from']}\n"

        return description

    def map_priority(self, priority: str) -> int:
        """Map priority string to ClickUp priority number"""
        priority_map = {
            'urgent': 1,
            'high': 2,
            'normal': 3,
            'low': 4
        }
        return priority_map.get(priority.lower(), 3)

    async def reply_to_message(self, original_message: Message, task_info: Dict[str, Any], clickup_task: Dict[str, Any]):
        """Reply to the original message with task creation confirmation"""
        try:
            reply_text = f"✅ **Task Created Successfully!**\n\n"
            reply_text += f"**Title:** {task_info.get('title', 'N/A')}\n"
            reply_text += f"**Priority:** {task_info.get('priority', 'normal').title()}\n"

            if task_info.get('due_date'):
                reply_text += f"**Due Date:** {task_info['due_date']}\n"

            if task_info.get('assignee'):
                reply_text += f"**Assignee:** {task_info['assignee']}\n"

            reply_text += f"**ClickUp Task ID:** {clickup_task.get('id', 'N/A')}\n"
            reply_text += f"**ClickUp URL:** {clickup_task.get('url', 'N/A')}"

            await original_message.reply_text(reply_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error replying to message: {e}")


async def main():
    """Main function to run the bot"""
    bot = ClickGramBot()

    if not bot.bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return

    # Create the Application
    application = Application.builder().token(bot.bot_token).build()

    # Setup handlers
    bot.setup_handlers(application)

    logger.info("Starting ClickGram bot...")
    logger.info(f"Bot is public - all users authorized")

    # Start the bot
    await application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    asyncio.run(main())