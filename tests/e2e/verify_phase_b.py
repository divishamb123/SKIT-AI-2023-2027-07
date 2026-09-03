import asyncio
import httpx

API_GATEWAY_URL = "http://localhost:8001/api/v1"
AUTH_SERVICE_URL = "http://localhost:8001/api/v1/auth"  # Gateway proxied


async def main():
    async with httpx.AsyncClient() as client:
        # 1. Register and Login to get JWT
        print("Registering user...")
        await client.post(
            f"{AUTH_SERVICE_URL}/register",
            json={
                "name": "Phase B User",
                "email": "phaseb_user@test.com",
                "password": "StrongPassword123!",
            },
        )

        print("Logging in...")
        login_res = await client.post(
            f"{AUTH_SERVICE_URL}/login",
            json={"email": "phaseb_user@test.com", "password": "StrongPassword123!"},
        )
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Upload Image
        print("Uploading Image to POST /api/v1/detect/image...")
        dummy_image = b"fake_image_data"
        files = {"file": ("test.webp", dummy_image, "image/webp")}

        # We need to simulate magic bytes for webp otherwise python-magic rejects it
        # WebP magic bytes: 52 49 46 46 (RIFF), 57 45 42 50 (WEBP)
        webp_magic = b"RIFF\x14\x00\x00\x00WEBPVP8 \x08\x00\x00\x00"
        files = {"file": ("test.webp", webp_magic, "image/webp")}

        upload_res = await client.post(
            f"{API_GATEWAY_URL}/detect/image", headers=headers, files=files
        )

        if upload_res.status_code != 202:
            print(f"Upload Failed: {upload_res.status_code} {upload_res.text}")
            exit(1)

        data = upload_res.json()
        job_id = data["job_id"]
        print(f"Upload Success! Job ID: {job_id}")

        # 3. Verify GET /api/v1/jobs/{id}
        print(f"Polling Job Status GET /api/v1/jobs/{job_id}...")
        job_res = await client.get(f"{API_GATEWAY_URL}/jobs/{job_id}", headers=headers)
        if job_res.status_code != 200:
            print(f"Job Poll Failed: {job_res.status_code} {job_res.text}")
            exit(1)

        job_data = job_res.json()
        print(
            f"Job Found! Status: {job_data['status']}, Modality: {job_data['modality']}"
        )

        if job_data["status"] == "queued":
            print("\nALL PHASE B RUNTIME VERIFICATIONS PASSED!")
        else:
            print(f"Unexpected status: {job_data['status']}")
            exit(1)


if __name__ == "__main__":
    asyncio.run(main())
