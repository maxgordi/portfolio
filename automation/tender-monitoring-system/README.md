# Tender Monitoring System

## Overview
An automated system for monitoring Ukrainian tender platforms for specific keywords and opportunities. The system continuously scans multiple tender websites, processes the information, and sends notifications via Telegram when relevant tenders are found.

## Features
- Multi-platform monitoring
- Real-time tender tracking
- Intelligent keyword matching
- Telegram notifications
- Duplicate detection
- Error handling and retry logic
- Configurable monitoring intervals
- Custom search patterns

## Supported Platforms
Platforms are configured in `config/platforms.json` (URL plus CSS selectors per site).
The file ships with placeholder entries: replace them with the tender sites you want to monitor.

## Technologies Used
- Python 3.8+
- Selenium WebDriver
- BeautifulSoup4
- Telegram Bot API
- Chrome WebDriver
- Async I/O
- Regular Expressions

## Prerequisites
- Python 3.8 or higher
- Google Chrome browser
- ChromeDriver matching your Chrome version
- Telegram Bot Token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tender-monitoring-system.git
cd tender-monitoring-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
- Copy `.env.example` to `.env`
- Add your Telegram bot token and chat ID
- Adjust monitoring settings if needed

## Configuration

### Environment Variables
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
MONITORING_INTERVAL=30  # seconds
MESSAGE_EXPIRATION=180  # seconds
```

### Keywords
Keywords can be configured in `config/keywords.json`:
```json
{
  "category_a": [
    "example keyword one",
    "example keyword two",
    ...
  ],
  "category_b": [
    "sample term",
    ...
  ]
}
```

## Usage

1. Start the monitoring system:
```bash
python src/tender_monitor.py
```

2. The system will:
- Initialize connections to tender platforms
- Start monitoring for keywords
- Send notifications when matches are found
- Handle errors and reconnections automatically

## Project Structure
```
tender-monitoring-system/
├── src/
│   ├── tender_monitor.py     # Main script
│   ├── platform_handlers/    # Platform-specific handlers
│   ├── notification.py       # Telegram notification
│   └── utils.py             # Utility functions
├── config/
│   ├── keywords.json        # Search keywords
│   └── platforms.json       # Platform configurations
├── requirements.txt
└── README.md
```

## Notification Format
```
🔍 New Tender Found!
Platform: [Platform Name]
Title: [Tender Title]
Keywords: [Matched Keywords]
Link: [Tender URL]
```

## Error Handling
- Connection issues
- Platform availability
- Rate limiting
- Invalid responses
- Duplicate detection

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License - see the [LICENSE](LICENSE) file for details
