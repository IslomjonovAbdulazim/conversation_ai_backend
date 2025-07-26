import jwt
import requests
import logging
from typing import Optional, Dict
from app.config import settings

logger = logging.getLogger(__name__)


class AppleAuthService:
    def __init__(self):
        self.apple_keys_url = "https://appleid.apple.com/auth/keys"
        self._public_keys = None

    async def get_apple_public_keys(self) -> Dict:
        """Get Apple's public keys for token verification"""
        if self._public_keys is None:
            try:
                response = requests.get(self.apple_keys_url, timeout=10)
                response.raise_for_status()
                self._public_keys = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch Apple public keys: {str(e)}")
                return {}
        return self._public_keys

    async def verify_identity_token(self, identity_token: str) -> Optional[Dict]:
        """Verify Apple identity token and return user data"""
        try:
            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(identity_token)
            key_id = unverified_header.get("kid")

            if not key_id:
                logger.error("No key ID found in Apple token header")
                return None

            # Get Apple's public keys
            keys_response = await self.get_apple_public_keys()
            keys = keys_response.get("keys", [])

            # Find the matching public key
            public_key = None
            for key in keys:
                if key.get("kid") == key_id:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                    break

            if not public_key:
                logger.error("No matching public key found for Apple token")
                return None


            # Verify and decode the token
            payload = jwt.decode(
                identity_token,
                public_key,
                algorithms=["RS256"],
                audience=settings.apple_team_id,  # Your app's bundle ID
                issuer="https://appleid.apple.com"
            )

            return payload

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
    """Main function to verify Apple identity token"""
    return await apple_auth_service.verify_identity_token(identity_token)