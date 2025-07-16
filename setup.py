import os
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def setup_telegram_session():
    """Setup Telegram session for the bot"""
    api_id = int(os.getenv('TELEGRAM_API_ID'))
    api_hash = os.getenv('TELEGRAM_API_HASH')
    session_name = os.getenv('BOT_SESSION_NAME', 'clickgram_bot')
    
    client = TelegramClient(session_name, api_id, api_hash)
    
    print("Setting up Telegram session...")
    await client.start()
    
    # Get information about the current user
    me = await client.get_me()
    print(f"Successfully logged in as: {me.first_name} (@{me.username})")
    
    # List available dialogs (chats/channels)
    print("\nAvailable chats/channels:")
    async for dialog in client.iter_dialogs():
        print(f"- {dialog.name} (ID: {dialog.id})")
    
    await client.disconnect()
    print("\nSetup complete! You can now use the bot.")

if __name__ == "__main__":
    asyncio.run(setup_telegram_session())
