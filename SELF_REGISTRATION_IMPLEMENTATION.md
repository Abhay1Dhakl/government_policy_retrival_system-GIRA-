# Self-Registration Feature Implementation

## Overview
Modified the authentication system from **admin-invite-only** to allow **user self-registration** with both email/password and Google OAuth options.

## Backend Changes

### 1. New User Registration Serializer
**File**: `gira-backend/src/users/serializers/user.py`

Added `UserRegistrationSerializer` with the following features:
- **Fields**: email, password, password_confirm, first_name, last_name
- **Validation**:
  - Password matching validation (password vs password_confirm)
  - Django's built-in password strength validation
  - Minimum length, common password checks, etc.
- **User Creation**: Creates user with `is_active=True` (no invitation needed)

```python
class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)
    password_confirm = serializers.CharField(write_only=True, required=True)
    
    def validate(self, attrs):
        # Password matching and strength validation
        ...
    
    def create(self, validated_data):
        # Creates active user immediately
        user.is_active = True
        ...
```

### 2. Public Registration Endpoint
**File**: `gira-backend/src/users/views/user.py`

Added new `@action` method `register()`:
- **URL**: `POST /api/v1/users/register/`
- **Permissions**: `AllowAny` (no authentication required)
- **Response**: Returns JWT tokens immediately for auto-login
- **Data Structure**:
  ```json
  {
    "user": { ... },
    "tokens": {
      "access": "...",
      "refresh": "..."
    }
  }
  ```

### 3. Google OAuth Auto-Registration
**File**: `gira-backend/src/users/views/token.py`

Modified `oauth_callback()` method:
- **Before**: Returned 404 if user doesn't exist
- **After**: Auto-creates user from Google profile data
- **Extraction**: Gets first_name/last_name from Google's `given_name`, `family_name`, or splits `name`
- **Activation**: Sets `is_active=True` automatically

### 4. Import Path Fixes
**Files**: `gira-backend/src/users/views/*.py`

Updated imports from `src.mira.utils.response` to `src.gira.utils.response` to match renamed directories.

## Frontend Changes

### 1. Registration Page
**File**: `gira_frontend/src/app/register/page.tsx`

New registration page with:
- Form state management for all registration fields
- Password matching validation
- Error handling with backend validation messages
- Auto-login after successful registration
- Google OAuth integration
- Redirect to `/users` page after registration

### 2. Registration Form Component
**File**: `gira_frontend/src/components/auth/RegisterForm.tsx`

Beautiful registration form with:
- Fields: First Name, Last Name, Email, Password, Confirm Password
- Password visibility toggles for both password fields
- Google Sign-In button integration
- Real-time validation feedback
- Loading states
- Error message display (supports multi-line validation errors)
- Link to login page for existing users

### 3. Login Page Update
**File**: `gira_frontend/src/components/auth/LoginForm.tsx`

Added:
- "Sign Up" link at the bottom for new users
- Imported Next.js `Link` component

### 4. Auth Service Update
**File**: `gira_frontend/src/lib/auth.ts`

Added `register()` method:
- **Interface**: `RegisterData` with all registration fields
- **Response Interface**: `RegisterResponse` with user and tokens
- **Error Handling**: Preserves backend validation errors for display
- **Endpoint**: Calls `/api/auth/register`

### 5. API Route for Registration
**File**: `gira_frontend/src/app/api/auth/register/route.ts`

New Next.js API route:
- Proxies registration requests to Django backend
- Handles CSRF token acquisition
- Proper error forwarding with validation details
- CORS preflight support

## API Endpoints

### Registration Endpoint
```
POST /api/v1/users/register/

Request Body:
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "password_confirm": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe"
}

Response (201 Created):
{
  "status": 201,
  "message": "User registered successfully",
  "data": {
    "user": {
      "id": 1,
      "email": "user@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "role": "user",
      "is_active": true
    },
    "tokens": {
      "access": "eyJ0eXAiOiJKV1QiLCJhbGc...",
      "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
    }
  }
}
```

### Google OAuth (Updated)
```
POST /api/v1/token/oauth/callback/

Behavior:
- If user exists: Returns tokens (as before)
- If user doesn't exist: Auto-creates user from Google profile, returns tokens
```

## User Flow

### Email/Password Registration
1. User visits `/register`
2. Fills out registration form (email, password, name)
3. Submits form → frontend validates password match
4. Backend validates password strength and creates user
5. Returns JWT tokens immediately
6. User is auto-logged in and redirected to `/users` to complete profile

### Google OAuth Registration
1. User clicks "Sign in with Google" on `/login` or `/register`
2. Completes Google authentication
3. Backend checks if user exists:
   - **Exists**: Returns tokens (normal login)
   - **New**: Auto-creates user from Google profile, returns tokens
4. User is auto-logged in
5. Redirects based on `has_details`:
   - `false` → `/users` (complete profile)
   - `true` → `/chat` (use app)

## Security Features

### Password Validation
- Minimum length (Django default: 8 characters)
- Not too common (checks against common password list)
- Not too similar to user information
- Not entirely numeric

### Django Password Validators
Uses Django's built-in validators:
- `UserAttributeSimilarityValidator`
- `MinimumLengthValidator`
- `CommonPasswordValidator`
- `NumericPasswordValidator`

### Authentication
- JWT tokens with access/refresh pattern
- Tokens stored in localStorage
- Backend validates all user input
- CSRF protection on all endpoints

## Testing the Feature

### Test Self-Registration
1. Start the backend: `docker compose up`
2. Visit: `http://localhost:3000/register`
3. Fill out the form:
   - Email: test@example.com
   - Password: SecurePass123!
   - Confirm Password: SecurePass123!
   - First Name: Test
   - Last Name: User
4. Click "Create Account"
5. Should redirect to `/users` page with auth token

### Test Google OAuth Registration
1. Visit: `http://localhost:3000/login`
2. Click "Sign in with Google"
3. Use a Google account that hasn't registered before
4. Should auto-create user and redirect to `/users` or `/chat`

### Test Validation
1. Try mismatched passwords → Should show error
2. Try weak password (e.g., "12345678") → Should show validation error
3. Try existing email → Should show "user already exists" error

## Migration Notes

### Breaking Changes
- None - this is additive functionality
- Old admin-invite flow still works via `POST /api/v1/users/` (admin-only)

### Database Changes
- No migrations needed
- Uses existing User model

### Backward Compatibility
- ✅ Admin can still create users via `/api/v1/users/` endpoint
- ✅ Existing users can log in normally
- ✅ Google OAuth works for both existing and new users
- ✅ Invitation emails still work if admin creates user

## Future Enhancements

### Recommended Additions
1. **Email Verification**: Send verification email after registration
2. **Password Reset**: Implement "Forgot Password" functionality
3. **Rate Limiting**: Prevent registration spam
4. **Captcha**: Add reCAPTCHA to prevent bots
5. **Terms of Service**: Add ToS acceptance checkbox
6. **Email Uniqueness Error**: Better error message for duplicate emails
7. **Username Support**: Optional username in addition to email

### Security Enhancements
1. **Account Lockout**: After failed login attempts
2. **Password History**: Prevent reusing old passwords
3. **Two-Factor Authentication**: Optional 2FA
4. **Session Management**: View/revoke active sessions
5. **Audit Logging**: Track registration attempts

## Files Modified

### Backend
- ✅ `gira-backend/src/users/serializers/user.py` - Added UserRegistrationSerializer
- ✅ `gira-backend/src/users/views/user.py` - Added register() endpoint
- ✅ `gira-backend/src/users/views/token.py` - Modified oauth_callback()
- ✅ `gira-backend/src/users/views/password.py` - Fixed import path

### Frontend
- ✅ `gira_frontend/src/app/register/page.tsx` - New registration page
- ✅ `gira_frontend/src/components/auth/RegisterForm.tsx` - New registration form
- ✅ `gira_frontend/src/components/auth/LoginForm.tsx` - Added sign-up link
- ✅ `gira_frontend/src/lib/auth.ts` - Added register() method
- ✅ `gira_frontend/src/app/api/auth/register/route.ts` - New API route

## Summary

The authentication system now supports:
- ✅ User self-registration with email/password
- ✅ Automatic user creation via Google OAuth
- ✅ Immediate login after registration (JWT tokens)
- ✅ Password strength validation
- ✅ Beautiful registration UI matching login page
- ✅ Proper error handling and validation messages
- ✅ Backward compatibility with admin-invite flow

Users can now register themselves without admin intervention, while administrators can still manually invite users if needed.
