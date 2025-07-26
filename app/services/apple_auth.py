import jwt
import requests
import json
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict

from app.config import settings

logger = logging.getLogger(__name__)

# Apple's public keys endpoint
APPLE_PUBLIC_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"


class AppleAuthService:

    def __init__(self):
        self.public_keys = None
        self.keys_last_updated = None

    async def get_apple_public_keys(self):
        """Fetch Apple's public keys for token verification"""
        try:
            # Cache keys for 1 hour
            if (self.public_keys and self.keys_last_updated and
                    datetime.utcnow() - self.keys_last_updated < timedelta(hours=1)):
                return self.public_keys

            response = requests.get(APPLE_PUBLIC_KEYS_URL, timeout=10)
            response.raise_for_status()

            self.public_keys = response.json()
            self.keys_last_updated = datetime.utcnow()

            return self.public_keys

        except Exception as e:
            logger.error(f"Failed to fetch Apple public keys: {str(e)}")
            return None

    def get_key_by_kid(self, kid: str, keys_data: dict):
        """Find the correct public key by key ID"""
        for key in keys_data.get("keys", []):
            if key.get("kid") == kid:
                return key
        return None

    async def verify_identity_token(self, identity_token: str) -> Optional[Dict]:
        """
        Verify Apple identity token and return user data
        """
        try:
            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(identity_token)
            kid = unverified_header.get("kid")

            if not kid:
                logger.error("No key ID found in token header")
                return None

            # Get Apple's public keys
            keys_data = await self.get_apple_public_keys()
            if not keys_data:
                logger.error("Failed to get Apple public keys")
                return None

            # Find the correct key
            public_key_data = self.get_key_by_kid(kid, keys_data)
            if not public_key_data:
                logger.error(f"Public key not found for kid: {kid}")
                return None

            # Convert JWK to PEM format
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(public_key_data))

            # Verify and decode the token
            decoded_token = jwt.decode(
                identity_token,
                public_key,
                algorithms=["RS256"],
                audience="com.azimislom.yodlaApp",
                issuer=APPLE_ISSUER
            )

            # Extract user information
            user_data = {
                "email": decoded_token.get("email"),
                "email_verified": decoded_token.get("email_verified", False),
                "sub": decoded_token.get("sub"),  # Apple user identifier
                "aud": decoded_token.get("aud"),
                "iss": decoded_token.get("iss"),
                "exp": decoded_token.get("exp"),
                "iat": decoded_token.get("iat")
            }

            # Validate required fields
            if not user_data.get("email"):
                logger.error("No email found in Apple token")
                return None

            logger.info(f"Apple token verified successfully for user: {user_data['email']}")
            return user_data

        except jwt.ExpiredSignatureError:
            logger.error("Apple identity token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid Apple identity token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Apple token verification failed: {str(e)}")
            return None


# Create service instance
apple_auth_service = AppleAuthService()


async def verify_apple_token(identity_token: str) -> Optional[Dict]:
    """
    Main function to verify Apple identity token
    """
    return await apple_auth_service.verify_identity_token(identity_token)


def create_client_secret():
    """
    Create client secret for server-to-server Apple authentication
    (Optional - only needed for some advanced features)
    """
    try:
        # Load private key
        with open(settings.apple_private_key_path, "r") as f:
            private_key_content = f.read()

        private_key = serialization.load_pem_private_key(
            private_key_content.encode(),
            password=None
        )

        # Create JWT payload
        now = datetime.utcnow()
        payload = {
            "iss": settings.apple_team_id,
            "iat": now,
            "exp": now + timedelta(minutes=5),  # 5 minutes expiry
            "aud": "https://appleid.apple.com",
            "sub": settings.apple_team_id  # Your app's bundle ID
        }

        # Create JWT
        client_secret = jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers={"kid": settings.apple_key_id}
        )

        return client_secret

    except Exception as e:
        logger.error(f"Failed to create Apple client secret: {str(e)}")
        return None