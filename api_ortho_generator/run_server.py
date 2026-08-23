import uvicorn

if __name__ == "__main__":
    print("🚀 Starting OrthoGenerator 2D Orthophoto Microservice...")
    print("📖 Swagger API Docs: http://localhost:8001/docs")
    print("📊 Health Check:    http://localhost:8001/api/v1/health")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=False)
