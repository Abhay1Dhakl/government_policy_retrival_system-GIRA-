# Quick Test Guide - Self-Registration Feature

## Prerequisites
- Docker containers running: `docker compose up`
- Backend accessible at: `http://localhost:8082`
- Frontend accessible at: `http://localhost:3000`

## Test 1: Email/Password Registration

### Steps:
1. Open browser to `http://localhost:3000/register`
2. Fill in the form:
   - **First Name**: Test
   - **Last Name**: User
   - **Email**: testuser@example.com
   - **Password**: TestPassword123!
   - **Confirm Password**: TestPassword123!
3. Click "Create Account"

### Expected Results:
- ✅ Form submits successfully
- ✅ You receive JWT access and refresh tokens
- ✅ Redirected to `/users` page (profile completion)
- ✅ You are logged in automatically

### Backend Verification:
```bash
# Check that user was created in database
docker exec -it gira-postgres psql -U postgres -d gira_db -c "SELECT id, email, first_name, last_name, is_active FROM users_user WHERE email='testuser@example.com';"
```

Expected output:
```
 id |         email          | first_name | last_name | is_active 
----+------------------------+------------+-----------+-----------
  X | testuser@example.com   | Test       | User      | t
```

## Test 2: Password Validation

### Test 2a: Passwords Don't Match
1. Go to `http://localhost:3000/register`
2. Fill form with:
   - Password: TestPassword123!
   - Confirm Password: DifferentPassword123!
3. Click "Create Account"

**Expected**: Error message "Passwords do not match"

### Test 2b: Weak Password
1. Go to `http://localhost:3000/register`
2. Fill form with:
   - Password: 12345678
   - Confirm Password: 12345678
3. Click "Create Account"

**Expected**: Error message about password being too common/weak

### Test 2c: Short Password
1. Go to `http://localhost:3000/register`
2. Fill form with:
   - Password: Test1!
   - Confirm Password: Test1!
3. Click "Create Account"

**Expected**: Error message "Password must be at least 8 characters"

## Test 3: Duplicate Email

1. Register a user with email `duplicate@example.com`
2. Try to register again with the same email
3. **Expected**: Error message about email already existing

## Test 4: Google OAuth Auto-Registration

### Steps:
1. Go to `http://localhost:3000/login`
2. Click the "Sign in with Google" button
3. Sign in with a Google account that has **never been used** in the system

### Expected Results:
- ✅ Google authentication succeeds
- ✅ User is automatically created in the database
- ✅ First and last names are extracted from Google profile
- ✅ User receives JWT tokens
- ✅ Redirected to `/users` (if name missing) or `/chat` (if complete)

### Backend Verification:
```bash
# Check that Google user was auto-created
docker exec -it gira-postgres psql -U postgres -d gira_db -c "SELECT id, email, first_name, last_name, is_active, password FROM users_user WHERE email='<your-google-email>';"
```

Expected output:
```
 id |       email        | first_name | last_name | is_active | password 
----+--------------------+------------+-----------+-----------+----------
  X | you@gmail.com      | Your       | Name      | t         | 
```

Note: Password field should be empty/null for OAuth users

## Test 5: Registration → Profile Completion Flow

### Steps:
1. Register new user via email/password
2. After redirect to `/users`, complete profile:
   - Add/verify first name and last name
3. Submit profile

### Expected Results:
- ✅ Profile saved successfully
- ✅ User redirected to `/chat` page
- ✅ User can now use the application

## Test 6: Login After Registration

### Steps:
1. Register user: `newuser@example.com` / `SecurePass123!`
2. Logout (or clear localStorage)
3. Go to `http://localhost:3000/login`
4. Login with: `newuser@example.com` / `SecurePass123!`

### Expected Results:
- ✅ Login succeeds
- ✅ Receives access/refresh tokens
- ✅ Redirected appropriately based on profile completion

## Test 7: UI/UX Elements

### Registration Page
- ✅ All form fields are present and labeled
- ✅ Password fields have show/hide toggle (eye icon)
- ✅ "Sign Up" button is disabled until all fields filled
- ✅ Google Sign-In button appears above the form
- ✅ "Already have an account? Sign In" link at bottom

### Login Page
- ✅ "Don't have an account? Sign Up" link appears
- ✅ Clicking link navigates to `/register`

## Common Issues & Troubleshooting

### Issue: "Network Error" or "Failed to fetch"
**Solution**: Check that Docker containers are running:
```bash
docker compose ps
```

All services should be "Up". If not:
```bash
docker compose down
docker compose up
```

### Issue: CORS errors in browser console
**Solution**: Check that API_BASE_URL is correct in frontend:
```bash
# Check .env.local
cat gira_frontend/.env.local | grep NEXT_PUBLIC_API_BASE_URL
```

Should be: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8082/api/v1`

### Issue: "CSRF token missing"
**Solution**: Backend needs to be accessible. Check:
```bash
curl http://localhost:8082/
```

Should return HTML (Django welcome page).

### Issue: Google Sign-In button doesn't appear
**Solution**: Check Google Client ID in frontend:
```bash
cat gira_frontend/.env.local | grep GOOGLE_CLIENT_ID
```

Should be set to your Google OAuth client ID.

### Issue: Password validation errors unclear
**Check backend logs**:
```bash
docker compose logs gira-backend | grep -i error
```

## API Testing (Using curl)

### Test Registration Endpoint Directly
```bash
curl -X POST http://localhost:8082/api/v1/users/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "apitest@example.com",
    "password": "ApiTestPass123!",
    "password_confirm": "ApiTestPass123!",
    "first_name": "API",
    "last_name": "Test"
  }'
```

**Expected Response** (201 Created):
```json
{
  "status": 201,
  "message": "User registered successfully",
  "data": {
    "user": {
      "id": ...,
      "email": "apitest@example.com",
      "first_name": "API",
      "last_name": "Test",
      "is_active": true
    },
    "tokens": {
      "access": "eyJ0eXAiOiJKV1Qi...",
      "refresh": "eyJ0eXAiOiJKV1Qi..."
    }
  }
}
```

### Test Login After Registration
```bash
curl -X POST http://localhost:8082/api/v1/token/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "apitest@example.com",
    "password": "ApiTestPass123!"
  }'
```

**Expected**: Returns access and refresh tokens

## Database Queries for Verification

### Count total users
```bash
docker exec -it gira-postgres psql -U postgres -d gira_db -c "SELECT COUNT(*) FROM users_user;"
```

### List all active users
```bash
docker exec -it gira-postgres psql -U postgres -d gira_db -c "SELECT id, email, first_name, last_name, is_active, date_joined FROM users_user WHERE is_active=true ORDER BY date_joined DESC;"
```

### Check user registration method (password vs OAuth)
```bash
docker exec -it gira-postgres psql -U postgres -d gira_db -c "SELECT email, CASE WHEN password IS NULL OR password = '' THEN 'OAuth' ELSE 'Password' END as auth_method FROM users_user;"
```

## Success Criteria

All tests should pass:
- ✅ Users can register with email/password
- ✅ Password validation works correctly
- ✅ Users receive JWT tokens immediately after registration
- ✅ Google OAuth auto-creates users
- ✅ UI is responsive and user-friendly
- ✅ Error messages are clear and helpful
- ✅ Users are logged in automatically after registration
- ✅ Backend correctly validates all inputs
- ✅ Database records are created correctly

## Next Steps After Testing

If all tests pass:
1. Commit the changes to Git
2. Update AUTHENTICATION_DOCUMENTATION.md with new registration flow
3. Consider implementing email verification (optional)
4. Add rate limiting to prevent spam registrations
5. Deploy to production environment

If tests fail:
1. Check Docker container logs: `docker compose logs`
2. Check browser console for frontend errors
3. Verify .env.local configuration
4. Ensure all files were saved correctly
5. Try restarting containers: `docker compose restart`
