import os
import sys
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def get_channel_info():
    """Get channel information for configuration"""
    api_id = int(os.getenv('TELEGRAM_API_ID'))
    api_hash = os.getenv('TELEGRAM_API_HASH')
    session_name = os.getenv('BOT_SESSION_NAME', 'clickgram_bot')
    
    client = TelegramClient(session_name, api_id, api_hash)
    
    print("Getting channel information...")
    await client.start()
    
    if len(sys.argv) > 1:
        channel_input = sys.argv[1]
        try:
            # Try to get entity by username or ID
            if channel_input.startswith('@'):
                entity = await client.get_entity(channel_input)
            else:
                entity = await client.get_entity(int(channel_input))
            
            print(f"\nChannel Information:")
            print(f"ID: {entity.id}")
            print(f"Username: @{entity.username}" if hasattr(entity, 'username') and entity.username else "No username")
            
            # Get recent messages to test
            print(f"\nRecent messages (last 5):")
            async for message in client.iter_messages(entity, limit=5):
                print(f"- {message.date}: {message.text[:50]}..." if message.text else f"- {message.date}: [Media message]")
            
        except Exception as e:
            print(f"Error getting channel info: {e}")
    else:
        print("Usage: python channel_info.py <channel_username_or_id>")
        print("Example: python channel_info.py @my_channel")
        print("Example: python channel_info.py -1001234567890")
    
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(get_channel_info())
