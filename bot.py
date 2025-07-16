import os
import asyncio
import logging
from typing import Dict, Any, Optional
from telethon import TelegramClient, events
from telethon.tl.types import Message
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import json
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClickGramBot:
    def __init__(self):
        # Telegram configuration
        self.api_id = int(os.getenv('TELEGRAM_API_ID'))
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.session_name = os.getenv('BOT_SESSION_NAME', 'clickgram_bot')
        self.target_channel_id = os.getenv('TARGET_CHANNEL_ID')
        
        # Gemini configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # ClickUp configuration
        self.clickup_api_key = os.getenv('CLICKUP_API_KEY')
        self.clickup_list_id = os.getenv('CLICKUP_LIST_ID')
        self.clickup_headers = {
            'Authorization': self.clickup_api_key,
            'Content-Type': 'application/json'
        }
        
        # Initialize Telegram client
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        
        # Task keywords that trigger processing
        self.task_keywords = [
            'task', 'todo', 'assignment', 'action item', 'deliverable',
            'deadline', 'due', 'complete', 'finish', 'work on', 'implement',
            'create', 'build', 'develop', 'fix', 'resolve', 'handle', 'done',
            'look at', 'asap', 'as soon'

        ]
    
    async def start(self):
        """Start the bot"""
        logger.info("Starting ClickGram bot...")
        
        # Connect to Telegram
        await self.client.start()
        logger.info("Connected to Telegram")
        
        # Verify target channel
        if self.target_channel_id:
            try:
                entity = await self.client.get_entity(int(self.target_channel_id))
                logger.info(f"Monitoring channel: {entity.username}")
            except Exception as e:
                logger.error(f"Error getting channel entity: {e}")
        
        # Register event handlers
        self.client.add_event_handler(self.handle_new_message, events.NewMessage)
        
        logger.info("Bot started successfully!")
        
        # Keep the bot running
        await self.client.run_until_disconnected()
    
    async def handle_new_message(self, event):
        """Handle new messages from the monitored channel"""
        try:
            message = event.message
            
            # Check if message is from target channel
            if self.target_channel_id and str(message.chat_id) != self.target_channel_id:
                return
            
            # Check if message contains task-related keywords
            if not self.contains_task_keywords(message.text):
                return
            
            logger.info(f"Processing potential task message: {message.text[:100]}...")
            
            # Process message with Gemini
            task_info = await self.process_message_with_gemini(message.text)
            
            if task_info and task_info.get('is_task'):
                # Create ClickUp task
                clickup_task = await self.create_clickup_task(task_info, message)
                
                if clickup_task:
                    logger.info(f"Created ClickUp task: {clickup_task['name']}")
                    
                    # Optional: Reply to the original message
                    await self.reply_to_message(message, task_info, clickup_task)
                else:
                    logger.error("Failed to create ClickUp task")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def contains_task_keywords(self, text: str) -> bool:
        """Check if message contains task-related keywords"""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.task_keywords)
    
    async def process_message_with_gemini(self, message_text: str) -> Optional[Dict[str, Any]]:
        """Process message with Gemini API to extract task information"""
        try:
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
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            try:
                task_info = json.loads(response.text.replace("```json", "").strip().strip("`"))
                return task_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini response as JSON: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message with Gemini: {e}")
            return None
    
    async def create_clickup_task(self, task_info: Dict[str, Any], original_message: Message) -> Optional[Dict[str, Any]]:
        """Create a task in ClickUp"""
        try:
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
                except ValueError:
                    logger.warning(f"Invalid due date format: {task_info['due_date']}")
            
            # Add time estimate if specified
            if task_info.get('estimated_hours'):
                task_data['time_estimate'] = task_info['estimated_hours'] * 3600000  # Convert to milliseconds
            
            # Create task via ClickUp API
            url = f"https://api.clickup.com/api/v2/list/{self.clickup_list_id}/task"
            
            response = requests.post(
                url,
                headers=self.clickup_headers,
                json=task_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"ClickUp API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating ClickUp task: {e}")
            return None
    
    def format_task_description(self, task_info: Dict[str, Any], original_message: Message) -> str:
        """Format task description with additional context"""
        description = task_info.get('description', '')
        
        # Add original message context
        description += f"\n\n**Original Message:**\n{original_message.text}"
        
        # Add message metadata
        description += f"\n\n**Message Details:**\n"
        description += f"- Date: {original_message.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        description += f"- Message ID: {original_message.id}\n"
        
        if original_message.sender:
            description += f"- From: {original_message.sender.first_name or 'Unknown'}\n"
        
        if task_info.get('assignee'):
            description += f"- Assigned to: {task_info['assignee']}\n"
        
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
            
            await self.client.send_message(
                original_message.chat_id,
                reply_text,
                reply_to=original_message.id
            )
            
        except Exception as e:
            logger.error(f"Error replying to message: {e}")
    
    async def stop(self):
        """Stop the bot"""
        logger.info("Stopping ClickGram bot...")
        await self.client.disconnect()

async def main():
    bot = ClickGramBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
