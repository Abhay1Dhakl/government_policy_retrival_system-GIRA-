// /app/api/transcribe/route.ts
import { NextResponse } from 'next/server';

export const runtime = 'edge'; // ensures compatibility

export async function POST(req: Request) {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: 'OpenAI API key missing' }, { status: 500 });
    }

    // Parse FormData
    const formData = await req.formData();
    const file = formData.get('file') as Blob | null;

    if (!file) {
      return NextResponse.json({ error: 'No audio file uploaded' }, { status: 400 });
    }

    // Convert Blob to Buffer
    const arrayBuffer = await file.arrayBuffer();
    const buffer = new Uint8Array(arrayBuffer);

    // Create FormData to send to OpenAI
    const payload = new FormData();
    payload.append('file', new Blob([buffer]), 'recording.webm');
    payload.append('model', 'whisper-1');

    const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: payload as any,
    });

    const transcription = await response.json();

    return NextResponse.json(transcription);
  } catch (err: any) {
    console.error('Transcription API error:', err);
    return NextResponse.json({ error: err.message || 'Transcription failed' }, { status: 500 });
  }
}
