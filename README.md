# ClickGram Bot 🤖

A sophisticated public Telegram bot that creates ClickUp tasks from various content types using Google Gemini AI for intelligent content analysis and task extraction.

## ✨ Features

### 🎯 Core Functionality
- **Multi-Format Processing**: Handle text messages, photos, documents, audio, video, and voice messages
- **AI-Powered Analysis**: Uses Google Gemini AI to extract task information from content
- **Interactive Task Creation**: Guided conversation workflow with inline keyboards
- **ClickUp Integration**: Seamlessly create tasks in your ClickUp workspace

### 🛡️ Advanced Features
- **Public Access**: No user authorization required - available to everyone
- **Environment Validation**: Comprehensive configuration and connectivity checks
- **Health Monitoring**: Real-time bot health status and uptime tracking
- **Error Handling**: Robust circuit breaker pattern with retry logic
- **Logging**: 5-day rotating error logs with structured monitoring

### 🎮 Interactive Commands
- **Task Management**: Create, search, update, and view tasks
- **Priority Selection**: Inline keyboard for quick priority assignment
- **Multi-step Conversations**: Stateful dialogues for complex task creation
- **Validation Commands**: Environment and connectivity validation tools

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Telegram Bot Token
- ClickUp API Key
- Google Gemini API Key

### Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   CLICKUP_API_KEY=pk_your_clickup_api_key_here
   CLICKUP_LIST_ID=your_clickup_list_id_here
   ```

3. **Run the bot**
   ```bash
   python bot.py
   ```

## 📋 Available Commands

### 🎯 Task Management Commands
| Command | Description | Usage |
|---------|-------------|-------|
| `/start` | Start the bot and show welcome message | `/start` |
| `/help` | Show help message with all commands | `/help` |
| `/createtask` | Start interactive task creation | `/createtask` |
| `/mytasks` | Show tasks assigned to current user | `/mytasks` |
| `/search <query>` | Search tasks by title | `/search bug fix` |
| `/updatetask <id> <status/priority> <value>` | Update task status or priority | `/updatetask 123 status in_progress` |

### 🎮 General Commands
| Command | Description | Usage |
|---------|-------------|-------|
| `/cancel` | Cancel active task creation | `/cancel` |

### 💬 Interactive Features
- **Priority Selection**: Choose from Urgent, High, Normal, or Low priority
- **Task Confirmation**: Review and confirm task details before creation
- **Multi-step Conversations**: Guided workflow for comprehensive task creation
- **Inline Keyboards**: Quick selections for priority and assignee

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | ✅ | Telegram bot token from @BotFather | `123456789:ABCdefGHIjklMNOpqrsTUVwxyz` |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key | `AIzaSyBcDeFgHiJkLmNoPqRsTuVwXyZ123456789` |
| `CLICKUP_API_KEY` | ✅ | ClickUp API key | `pk_123456789_abcdefg` |
| `CLICKUP_LIST_ID` | ✅ | ClickUp list ID for task creation | `123456789` |

### ClickUp Setup

1. **Get your ClickUp API Key**:
   - Go to ClickUp Settings → Apps → API
   - Generate a new API key
   - Copy the key starting with `pk_`

2. **Find your List ID**:
   - Navigate to your desired list in ClickUp
   - The URL will contain the list ID: `https://app.clickup.com/12345678/v/l/987654321`
   - The number after `/l/` is your list ID

### Google Gemini Setup

1. **Get your Gemini API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for your configuration

## 🔧 Advanced Features

### Environment Validation
The bot includes comprehensive environment validation:

- **Configuration Checks**: Validates all required environment variables
- **API Connectivity**: Tests connections to Telegram, Gemini, and ClickUp APIs
- **Health Monitoring**: Tracks bot uptime and service availability

### Error Handling & Resilience
- **Circuit Breakers**: Prevents cascading failures for external services
- **Retry Logic**: Exponential backoff for failed API calls
- **Structured Logging**: 5-day rotating logs with error monitoring
- **Graceful Degradation**: Continues operating with limited functionality during outages

### Security Features
- **Public Access**: Bot is available to all users without restrictions
- **Secure Configuration**: No sensitive data exposed in logs
- **API Key Protection**: Secure storage and usage of credentials

## 🚨 Troubleshooting

### Common Issues

#### 1. Bot Not Starting
```bash
Error: TELEGRAM_BOT_TOKEN not found in environment variables
```
**Solution**: Ensure your `.env` file is properly configured with the bot token.


#### 2. ClickUp API Errors
```bash
Error: ClickUp API returned status 401: Unauthorized
```
**Solution**: Verify your ClickUp API key and list ID are correct.

#### 3. Gemini API Errors
```bash
Error: Google Generative AI returned status 400
```
**Solution**: Check your Gemini API key and ensure it's properly configured.

#### 4. Permission Issues
```bash
Error: Permission denied when accessing logs directory
```
**Solution**: Ensure the bot has write permissions for creating the `logs` directory.

### Debug Mode
For detailed logging, check the `logs/` directory:
- Error logs are rotated every 5 days
- Structured JSON format for easy parsing
- Comprehensive error context and stack traces

### Health Checks
If you experience issues, please try again later or contact support.
- Environment variable status
- API connectivity
- Bot uptime
- Service availability

## 📊 Monitoring & Maintenance

### Log Management
- Automatic cleanup of logs older than 5 days
- Structured error logging with context
- Real-time error monitoring with circuit breakers

### Performance Optimization
- Connection pooling for API requests
- Asynchronous processing for better performance
- Memory-efficient conversation management

### Backup & Recovery
- Environment configuration validation on startup
- Graceful handling of API failures
- Automatic recovery from temporary outages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Telegram Bot API](https://core.telegram.org/bots/api) for the bot framework
- [ClickUp API](https://clickup.com/api) for task management integration
- [Google Gemini AI](https://ai.google.dev/) for intelligent content analysis
- [python-telegram-bot](https://python-telegram-bot.org/) for the excellent Python library

## 📞 Support

For support, please:
1. Check the troubleshooting section above
2. Review the logs for detailed error information
3. If issues persist, the bot may be experiencing temporary service interruptions
4. Create an issue in the repository with relevant details

---

**ClickGram Bot** - Transform your Telegram messages into ClickUp tasks with AI-powered intelligence! 🚀
