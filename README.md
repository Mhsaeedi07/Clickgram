# ClickGram Bot

A Telegram bot that monitors channels for task assignments, processes them with AI, and creates ClickUp tasks automatically.

## What it does

- Watches Telegram channels for messages containing tasks
- Uses Google Gemini AI to extract task details (title, priority, assignee, due date)
- Creates tasks in ClickUp with all the extracted information

## Quick Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get your API keys:
   - Telegram: [my.telegram.org](https://my.telegram.org)
   - Google Gemini: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - ClickUp: Settings → Apps → API

3. Update `.env` file with your keys

4. Run setup:
   ```bash
   python setup.py          # Setup Telegram session
   python clickup_test.py   # Test ClickUp and get list ID
   python gemini_test.py    # Test Gemini API
   ```

5. Start the bot:
   ```bash
   python bot.py
   ```

## Configuration

Update `.env` with your credentials:

```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
GEMINI_API_KEY=your_gemini_api_key
CLICKUP_API_KEY=your_clickup_api_key
CLICKUP_LIST_ID=your_clickup_list_id
TARGET_CHANNEL_ID=your_target_channel_id
```

## Example Usage

The bot detects messages like:
- "John, please implement user authentication by Friday. High priority."
- "TODO: Create unit tests for the API. Due next week."
- "Can someone fix the payment bug ASAP?"

And automatically creates ClickUp tasks with extracted details.
