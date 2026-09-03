import asyncio
import httpx

API_GATEWAY_URL = "http://localhost:8001/api/v1"
AUTH_SERVICE_URL = "http://localhost:8001/api/v1/auth"


async def main():
    async with httpx.AsyncClient() as client:
        # 1. Register and Login to get JWT
        print("Registering user...")
        # Ignore 400 if already registered
        await client.post(
            f"{AUTH_SERVICE_URL}/register",
            json={
                "name": "Phase C User",
                "email": "phasec_user@test.com",
                "password": "StrongPassword123!",
            },
        )

        print("Logging in...")
        login_res = await client.post(
            f"{AUTH_SERVICE_URL}/login",
            json={"email": "phasec_user@test.com", "password": "StrongPassword123!"},
        )
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Upload Image
        print("Uploading Image to POST /api/v1/detect/image...")
        webp_magic = b"RIFF\x14\x00\x00\x00WEBPVP8 \x08\x00\x00\x00"
        files = {"file": ("test_c.webp", webp_magic, "image/webp")}

        upload_res = await client.post(
            f"{API_GATEWAY_URL}/detect/image", headers=headers, files=files
        )
        if upload_res.status_code != 202:
            print(f"Upload Failed: {upload_res.status_code} {upload_res.text}")
            exit(1)

        job_id = upload_res.json()["job_id"]
        print(f"Upload Success! Job ID: {job_id}")

        # 3. Poll for state transitions
        states_seen = set()
        print(f"Polling Job Status GET /api/v1/jobs/{job_id}...")

        max_attempts = 20
        for i in range(max_attempts):
            job_res = await client.get(
                f"{API_GATEWAY_URL}/jobs/{job_id}", headers=headers
            )
            if job_res.status_code != 200:
                print(f"Job Poll Failed: {job_res.status_code} {job_res.text}")
                exit(1)

            status = job_res.json()["status"]
            if status not in states_seen:
                print(f"New state observed: {status}")
                states_seen.add(status)

            if status == "completed" or status == "failed":
                break

            await asyncio.sleep(1)

        print("\nVerification Results:")
        print(f"States observed: {states_seen}")

        if (
            "queued" in states_seen
            and "processing" in states_seen
            and "completed" in states_seen
        ):
            print("ALL PHASE C RUNTIME VERIFICATIONS PASSED!")
        else:
            print(
                "Failed to observe all expected states (queued -> processing -> completed)!"
            )
            exit(1)


if __name__ == "__main__":
    asyncio.run(main())
