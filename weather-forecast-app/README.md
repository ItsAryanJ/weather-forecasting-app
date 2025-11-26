# Weather Forecast App (SARIMAX)
Full-stack project with:
- FastAPI backend (reads CSV from backend/data/)
- React + Vite frontend (Plotly visualizations)

## How to run

### Backend
$ cd backend
$ python -m venv venv
$ source venv/bin/activate   # Windows: venv\Scripts\activate
$ pip install -r requirements.txt
$ uvicorn main:app --reload --host 0.0.0.0 --port 8000

### Frontend
$ cd frontend
$ npm install
$ npm run dev

API root: http://localhost:8000/api
Frontend dev (Vite): http://localhost:5173
