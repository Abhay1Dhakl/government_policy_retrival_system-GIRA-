# GIRA Authentication & SSO System Documentation

## Overview
The GIRA system implements a comprehensive authentication system with:
- **JWT-based authentication** using Django REST Framework SimpleJWT
- **Google OAuth 2.0 SSO** integration
- **Custom user management** with role-based access control
- **Token refresh mechanism** for seamless user experience

---

## Architecture Components

### 1. Backend (Django REST Framework)

#### User Model (`gira-backend/src/users/models.py`)
- **Custom User Model** extending `AbstractBaseUser` and `PermissionsMixin`
- **Fields:**
  - `email` (unique identifier, used as USERNAME_FIELD)
  - `password` (hashed using Django's password hasher)
  - `first_name`, `last_name`, `phone_number`
  - `country`, `city`, `address`, `zip_code`
  - `institution`
  - `role` (admin/user)
  - `is_active` (default: False, activated on first login or by admin)
  - `is_staff`, `is_superuser`
  - `created_at`, `updated_at`

- **Key Methods:**
  - `set_password()`: Hashes and stores password
  - `has_details`: Property that returns True if first_name and last_name are set

#### Authentication Flow

##### A. Traditional Email/Password Login

**Endpoint:** `POST /api/v1/token/`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Process:**
1. Frontend sends credentials to `/api/auth/login` (Next.js API route)
2. Next.js API route forwards to Django backend `/api/v1/token/`
3. Backend validates credentials using `UserTokenSerializer`
4. If valid, generates JWT tokens using `CustomTokenObtainPairSerializer`
5. Returns access token, refresh token, and `has_details` flag

**Response:**
```json
{
  "status": 200,
  "message": "Token generated successfully",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "has_details": true
  }
}
```

**Token Payload (Custom JWT Claims):**
```json
{
  "user_id": 1,
  "email": "user@example.com",
  "llm_model": {
    "name": "GPT-4",
    "model": "gpt-4",
    "api_key": "encrypted_key"
  },
  "country": "NP",
  "exp": 1234567890,
  "iat": 1234567890
}
```

##### B. Google OAuth SSO Login

**Endpoint:** `POST /api/v1/token/oauth/callback/`

**Flow:**
1. **Frontend Integration:**
   - Google Sign-In button rendered using Google Identity Services
   - Client ID: `NEXT_PUBLIC_GOOGLE_CLIENT_ID` from environment
   - Button configuration in `LoginForm.tsx`:
     ```typescript
     google.accounts.id.initialize({
       client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
       callback: async (response: any) => {
         await onGoogleSignIn(response.credential);
       },
     });
     ```

2. **Google Authentication:**
   - User clicks "Sign in with Google"
   - Google OAuth popup appears
   - User authenticates with Google
   - Google returns a JWT credential

3. **Backend Verification:**
   - Frontend sends Google JWT to Next.js API route `/api/auth/google`
   - Next.js forwards to Django `/api/v1/token/oauth/callback/`
   - Backend verifies token using Google's `id_token.verify_oauth2_token()`
   - Extracts user email from verified token

4. **User Lookup:**
   - Backend checks if user exists with that email
   - If user exists: Generates GIRA JWT tokens
   - If user doesn't exist: Returns 404 (user must be invited first)

5. **Token Generation:**
   - Same JWT token structure as email/password login
   - Sets `is_active = True` for the user

**Request:**
```json
{
  "token": "eyJhbGciOiJSUzI1NiIsImtpZCI6..." // Google JWT
}
```

**Response:**
```json
{
  "status": 200,
  "message": "Token generated successfully",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "has_details": true
  }
}
```

#### JWT Configuration (`gira-backend/src/gira/settings.py`)

```python
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(days=7),  # Configurable via env
    "REFRESH_TOKEN_LIFETIME": timedelta(days=30),
    "ROTATE_REFRESH_TOKENS": False,
    "BLACKLIST_AFTER_ROTATION": True,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
}
```

#### Token Refresh

**Endpoint:** `POST /api/v1/token/refresh/`

**Request:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Process:**
1. Validates refresh token
2. Extracts user_id from token
3. Generates new access token
4. Returns new access token (refresh token remains same)

**Response:**
```json
{
  "status": 200,
  "message": "Token refreshed successfully",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "has_details": true
  }
}
```

---

### 2. Frontend (Next.js + TypeScript)

#### AuthService (`gira_frontend/src/lib/auth.ts`)

**Key Methods:**

1. **login(credentials):**
   - Calls `/api/auth/login` (Next.js API route)
   - Stores access token, refresh token, and has_details in localStorage
   - Returns login response

2. **googleSignIn(credential):**
   - Calls `/api/auth/google` with Google JWT
   - Stores tokens in localStorage
   - Returns authentication response

3. **Token Management:**
   - `setToken()`: Stores access token in localStorage
   - `getToken()`: Retrieves access token
   - `setRefreshToken()`: Stores refresh token
   - `getRefreshToken()`: Retrieves refresh token
   - `removeToken()`: Clears all auth data (logout)

4. **Profile Completion:**
   - `setHasDetails()`: Stores whether user completed profile
   - `isProfileCompleted()`: Checks if user has completed profile
   - Used to route users to `/users` for profile completion or `/chat` for app

5. **API Request Helper:**
   - `getAuthHeader()`: Returns authorization header with Bearer token
   ```typescript
   {
     Authorization: `Bearer ${token}`
   }
   ```

#### Login Page (`gira_frontend/src/app/login/page.tsx`)

**Features:**
- Email/password form
- Google Sign-In button
- Loading states
- Error handling
- Automatic routing based on profile completion:
  - `has_details === true` → `/chat`
  - `has_details === false` → `/users` (profile completion)

#### API Routes (Next.js Server-Side)

1. **`/api/auth/login/route.ts`:**
   - Proxy to Django backend
   - Handles CORS and authentication
   - Returns standardized response

2. **`/api/auth/google/route.ts`:**
   - Receives Google OAuth credential
   - Forwards to Django `/token/oauth/callback/`
   - Returns authentication response

---

## Security Features

### 1. Password Security
- Passwords hashed using Django's `make_password()` (PBKDF2 algorithm)
- Password complexity requirements enforced by frontend
- No plain-text password storage

### 2. JWT Security
- HTTPS-only in production (enforced by CORS settings)
- Short-lived access tokens (7 days default)
- Longer-lived refresh tokens (30 days default)
- Tokens stored in localStorage (consider httpOnly cookies for enhanced security)

### 3. Google OAuth Security
- Token verification using Google's official library
- Client ID validation
- No password storage for OAuth users
- Email verification handled by Google

### 4. CORS Configuration
```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://gira.govinfo.com",
]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = [
    'authorization',
    'content-type',
    'x-csrftoken',
]
```

---

## User Registration Flow

### Admin-Initiated User Invite

1. **Admin creates user account:**
   - Admin uses backend API to create user with email
   - User account created with `is_active=False`
   - Invitation email sent to user

2. **User receives invitation:**
   - Email contains link to create password
   - Link includes token for validation

3. **Password Creation:**
   - User visits `/create-password` with token
   - Sets password
   - Account activated (`is_active=True`)

4. **First Login:**
   - User can now login with email/password or Google SSO
   - Redirected to profile completion if `has_details=False`
   - Redirected to chat if profile completed

---

## Profile Completion Flow

**Purpose:** Collect additional user information after initial authentication

**Endpoint:** Not explicitly shown, but inferred from `has_details` property

**Fields to Complete:**
- First Name
- Last Name
- Phone Number
- Country
- City
- Address
- Zip Code
- Institution

**Once completed:**
- `has_details` becomes `true`
- User can access main application (`/chat`)

---

## Authorization & Permissions

### Django REST Framework Permissions

1. **IsAuthenticated:**
   - Default for most endpoints
   - Requires valid JWT token in Authorization header

2. **IsAdminUser:**
   - Required for user management endpoints
   - Checks `is_staff=True` or `is_superuser=True`

3. **AllowAny:**
   - Used for login, OAuth callback endpoints
   - No authentication required

### Role-Based Access

**User Roles:**
- `admin`: Full system access, user management
- `user`: Standard user access

**Implementation:**
- Stored in `User.role` field
- Can be used for custom permissions
- JWT token includes role information

---

## Environment Variables

### Backend (.env.local)
```env
# Django Secret
SECRET_KEY=your-secret-key

# JWT Configuration
ACCESS_TOKEN_LIFETIME=7  # days
REFRESH_TOKEN_LIFETIME=30  # days

# Google OAuth
OAUTH_CLIENT_ID=your-google-client-id

# CORS
CORS_ORIGINS_DEV=http://localhost:3000
CORS_ORIGINS_PROD=https://gira.govinfo.com
```

### Frontend (.env.local)
```env
# API Base URL
NEXT_PUBLIC_API_BASE_URL=http://gira-backend:8082/api/v1

# Google OAuth
NEXT_PUBLIC_GOOGLE_CLIENT_ID=1069463773061-dblgrum0aq1se6ve1khjk0fdj6583179.apps.googleusercontent.com
```

---

## API Endpoints Summary

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/v1/token/` | POST | Email/password login | No |
| `/api/v1/token/refresh/` | POST | Refresh access token | No |
| `/api/v1/token/oauth/callback/` | POST | Google OAuth callback | No |
| `/api/v1/users/` | GET | List users | Yes (Admin) |
| `/api/v1/users/` | POST | Create user | Yes (Admin) |
| `/api/v1/users/{id}/` | GET | Get user details | Yes |
| `/api/v1/users/{id}/` | PUT/PATCH | Update user | Yes |

---

## Testing Authentication

### 1. Test Email/Password Login
```bash
curl -X POST http://localhost:8082/api/v1/token/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123"
  }'
```

### 2. Test Token Refresh
```bash
curl -X POST http://localhost:8082/api/v1/token/refresh/ \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your-refresh-token"
  }'
```

### 3. Test Authenticated Request
```bash
curl -X GET http://localhost:8082/api/v1/users/ \
  -H "Authorization: Bearer your-access-token"
```

---

## Common Issues & Troubleshooting

### 1. "Invalid or expired refresh token"
- **Cause:** Refresh token has expired or is invalid
- **Solution:** User must login again

### 2. "User not found or you're not invited"
- **Cause:** Google OAuth user doesn't exist in system
- **Solution:** Admin must create user account first

### 3. CORS errors
- **Cause:** Frontend origin not in CORS_ALLOWED_ORIGINS
- **Solution:** Add origin to Django settings

### 4. Google OAuth not working
- **Cause:** Client ID mismatch or not configured
- **Solution:** Verify `NEXT_PUBLIC_GOOGLE_CLIENT_ID` matches Google Console

---

## Security Best Practices

1. **Use HTTPS in production** - All authentication must use secure connections
2. **Rotate JWT secret keys** - Periodically update SECRET_KEY
3. **Implement rate limiting** - Prevent brute force attacks
4. **Add token blacklist** - Track invalidated tokens
5. **Use httpOnly cookies** - Consider moving tokens from localStorage to httpOnly cookies
6. **Implement 2FA** - Add two-factor authentication for enhanced security
7. **Log authentication events** - Monitor login attempts and failures
8. **Set short token expiration** - Balance security vs. user experience

---

## Future Enhancements

1. **Multi-factor Authentication (MFA)**
2. **OAuth providers** (Microsoft, GitHub, etc.)
3. **Password reset flow**
4. **Email verification**
5. **Session management dashboard**
6. **Security audit logs**
7. **Device management**
8. **Biometric authentication**

---

## References

- Django REST Framework: https://www.django-rest-framework.org/
- SimpleJWT: https://django-rest-framework-simplejwt.readthedocs.io/
- Google Identity Services: https://developers.google.com/identity/gsi/web
- Next.js Authentication: https://nextjs.org/docs/authentication
