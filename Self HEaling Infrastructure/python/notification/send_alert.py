#!/usr/bin/env python3

import os
import sys
import json
import yaml
import logging
import argparse
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'notifications.log'), mode='a')
    ]
)
logger = logging.getLogger('sentinel_notifications')

# Global variables
config = {}

# Load configuration
def load_config() -> Dict:
    """Load the configuration from the settings.yaml file"""
    global config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'settings.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return {}

# Format message based on severity and content
def format_message(severity: str, target: str, issue: str, action: str, details: Dict = None) -> Dict[str, str]:
    """Format notification message based on severity and content"""
    timestamp = datetime.now().isoformat()
    
    # Set emoji based on severity
    emoji_map = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢',
        'info': '🔵'
    }
    emoji = emoji_map.get(severity.lower(), '⚪')
    
    # Format plain text message
    text_message = f"{emoji} {severity.upper()} ALERT - {issue}\n\n" \
                  f"Target: {target}\n" \
                  f"Timestamp: {timestamp}\n" \
                  f"Action: {action}\n"
    
    if details:
        text_message += "\nDetails:\n"
        for key, value in details.items():
            text_message += f"  {key}: {value}\n"
    
    # Format HTML message
    color_map = {
        'critical': '#FF0000',
        'high': '#FF7F00',
        'medium': '#FFFF00',
        'low': '#00FF00',
        'info': '#0000FF'
    }
    color = color_map.get(severity.lower(), '#FFFFFF')
    
    html_message = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .alert {{ padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .{severity.lower()} {{ background-color: {color}; color: {'#FFFFFF' if severity.lower() in ['critical', 'high'] else '#000000'}; }}
        .details {{ margin-top: 15px; border-top: 1px solid #ccc; padding-top: 10px; }}
        .footer {{ margin-top: 20px; font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="alert {severity.lower()}">
        <h2>{emoji} {severity.upper()} ALERT - {issue}</h2>
        <p><strong>Target:</strong> {target}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Action:</strong> {action}</p>
    </div>
"""
    
    if details:
        html_message += """    <div class="details">
        <h3>Details:</h3>
        <ul>
"""
        for key, value in details.items():
            html_message += f"            <li><strong>{key}:</strong> {value}</li>\n"
        html_message += """        </ul>
    </div>
"""
    
    html_message += """    <div class="footer">
        <p>This is an automated message from SelfHealing Sentinel.</p>
    </div>
</body>
</html>
"""
    
    # Format JSON message for webhooks
    json_message = {
        'severity': severity,
        'target': target,
        'issue': issue,
        'action': action,
        'timestamp': timestamp,
        'details': details or {}
    }
    
    return {
        'text': text_message,
        'html': html_message,
        'json': json_message
    }

# Send notification via Slack
def send_slack_notification(message: Dict[str, str]) -> bool:
    """Send notification to Slack"""
    if not config.get('notifications', {}).get('channels', {}).get('slack', {}).get('enabled', False):
        logger.info("Slack notifications are disabled")
        return False
    
    slack_config = config.get('notifications', {}).get('channels', {}).get('slack', {})
    webhook_url = slack_config.get('webhook_url')
    if not webhook_url:
        logger.error("Slack webhook URL not configured")
        return False
    
    try:
        payload = {
            'text': message['text'],
            'username': slack_config.get('username', 'SelfHealing Sentinel'),
            'icon_emoji': slack_config.get('icon_emoji', ':robot_face:'),
            'channel': slack_config.get('channel', '#alerts')
        }
        
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            logger.info("Slack notification sent successfully")
            return True
        else:
            logger.error(f"Failed to send Slack notification: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Slack notification: {str(e)}")
        return False

# Send notification via email
def send_email_notification(message: Dict[str, str], recipients: List[str] = None) -> bool:
    """Send notification via email"""
    if not config.get('notifications', {}).get('channels', {}).get('email', {}).get('enabled', False):
        logger.info("Email notifications are disabled")
        return False
    
    email_config = config.get('notifications', {}).get('channels', {}).get('email', {})
    smtp_server = email_config.get('smtp_server')
    smtp_port = email_config.get('smtp_port', 587)
    smtp_username = email_config.get('smtp_username')
    smtp_password = email_config.get('smtp_password')
    sender = email_config.get('sender', 'sentinel@example.com')
    
    if not smtp_server or not smtp_username or not smtp_password:
        logger.error("Email configuration incomplete")
        return False
    
    if not recipients:
        recipients = email_config.get('recipients', [])
    
    if not recipients:
        logger.error("No email recipients specified")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"SelfHealing Sentinel Alert: {message['json']['issue']}"
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        
        # Attach parts
        part1 = MIMEText(message['text'], 'plain')
        part2 = MIMEText(message['html'], 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender, recipients, msg.as_string())
        
        logger.info(f"Email notification sent to {', '.join(recipients)}")
        return True
    except Exception as e:
        logger.error(f"Error sending email notification: {str(e)}")
        return False

# Send notification via webhook
def send_webhook_notification(message: Dict[str, str]) -> bool:
    """Send notification to a webhook endpoint"""
    if not config.get('notifications', {}).get('channels', {}).get('webhook', {}).get('enabled', False):
        logger.info("Webhook notifications are disabled")
        return False
    
    webhook_config = config.get('notifications', {}).get('channels', {}).get('webhook', {})
    webhook_url = webhook_config.get('url')
    if not webhook_url:
        logger.error("Webhook URL not configured")
        return False
    
    try:
        method = webhook_config.get('method', 'POST').upper()
        headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
        
        if method == 'POST':
            response = requests.post(webhook_url, json=message['json'], headers=headers)
        elif method == 'PUT':
            response = requests.put(webhook_url, json=message['json'], headers=headers)
        else:
            logger.error(f"Unsupported webhook method: {method}")
            return False
        
        if response.status_code in [200, 201, 202, 204]:
            logger.info(f"Webhook notification sent successfully: {response.status_code}")
            return True
        else:
            logger.error(f"Failed to send webhook notification: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending webhook notification: {str(e)}")
        return False

# Send notification to all enabled channels
def send_notification(severity: str, target: str, issue: str, action: str, details: Dict = None) -> Dict[str, bool]:
    """Send notification to all enabled channels"""
    # Format message
    message = format_message(severity, target, issue, action, details)
    
    # Send to all enabled channels
    results = {}
    
    # Slack
    if config.get('notifications', {}).get('channels', {}).get('slack', {}).get('enabled', False):
        results['slack'] = send_slack_notification(message)
    
    # Email
    if config.get('notifications', {}).get('channels', {}).get('email', {}).get('enabled', False):
        results['email'] = send_email_notification(message)
    
    # Webhook
    if config.get('notifications', {}).get('channels', {}).get('webhook', {}).get('enabled', False):
        results['webhook'] = send_webhook_notification(message)
    
    # Log results
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    if total_count > 0:
        logger.info(f"Notification sent to {success_count}/{total_count} channels")
    else:
        logger.warning("No notification channels enabled")
    
    return results

# Main function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Send alerts from SelfHealing Sentinel')
    parser.add_argument('--severity', required=True, choices=['critical', 'high', 'medium', 'low', 'info'], help='Alert severity')
    parser.add_argument('--target', required=True, help='Target of the alert (e.g., node name, pod name)')
    parser.add_argument('--issue', required=True, help='Issue description')
    parser.add_argument('--action', required=True, help='Action taken or recommended')
    parser.add_argument('--details', help='JSON string with additional details')
    args = parser.parse_args()
    
    # Load configuration
    load_config()
    
    # Parse details if provided
    details = None
    if args.details:
        try:
            details = json.loads(args.details)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse details JSON: {args.details}")
            details = {'raw_details': args.details}
    
    # Send notification
    results = send_notification(args.severity, args.target, args.issue, args.action, details)
    
    # Exit with success if at least one channel succeeded
    if any(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

# Entry point
if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    main()