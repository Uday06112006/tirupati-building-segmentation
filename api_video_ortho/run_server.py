import uvicorn

if __name__ == "__main__":
    print("🚀 Starting OrthoGenerator Video-to-Frame Microservice...")
    print("📖 Swagger API Docs: http://localhost:8000/docs")
    print("📊 Health Check:    http://localhost:8000/api/v1/health")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
