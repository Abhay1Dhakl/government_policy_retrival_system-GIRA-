// "use client";

// import {jwtDecode} from "jwt-decode";

// export const getUserIdFromToken = (): string | null => {
//   if (typeof window === "undefined") return null; // Safe for SSR

//   const token = localStorage.getItem("token");
//   if (!token) return null;

//   try {
//     const decoded = jwtDecode<{ user_id: string }>(token);
//     return decoded.user_id;
//   } catch (err) {
//     console.error("Failed to decode token:", err);
//     return null;
//   }
// };