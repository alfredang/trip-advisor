# 🧳 AI Trip Planner (Multi-Agent System)

An interactive Streamlit web app that uses multiple specialized AI agents to create comprehensive travel plans.

🔗 **[Live Demo](https://tripplannerpy-hftrotqj67bccfezasz7iz.streamlit.app/)**

## Features

- **Interactive UI** - Input your destination, trip duration, budget, and special preferences
- **Multi-Agent Architecture** - Four specialized AI agents work together to create your perfect trip

### AI Agents

| Agent | Role |
|-------|------|
| 🧠 **Planner Agent** | Creates day-by-day itineraries |
| 💰 **Budget Agent** | Estimates and tracks costs |
| 🍣 **Local Guide Agent** | Recommends food & local tips |
| 🔍 **Research Agent** | Fetches current travel updates (used sparingly) |
| ✈️ **Travel Agent** | Orchestrates all agents and produces the final plan |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alfredang/trip-advisor.git
   cd trip-advisor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. Run the app:
   ```bash
   streamlit run trip_planner.py
   ```

## Usage

1. Enter your **destination** (default: Tokyo)
2. Set the **number of days** (default: 5)
3. Specify your **budget** in USD (default: $2,000)
4. Add any **special preferences** (optional)
5. Click **"Generate Trip Plan"**

The app will display your personalized trip plan with:
- 📋 Day-by-day itinerary
- 💰 Budget breakdown
- 🍣 Local recommendations and tips
- 🔍 Current travel updates (when available)
- 📥 Download option for your trip plan

## Tech Stack

- **Streamlit** - Web interface
- **OpenAI Agents SDK** - Multi-agent orchestration
- **Gemini 2.0 Flash** - LLM backend
- **Tavily** - Real-time web search
- **Pydantic** - Data validation

## License

MIT
