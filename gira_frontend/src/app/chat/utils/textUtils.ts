"use client";

export function extractKeyPhrasesForHighlight(text: string, maxLength: number = 50): string[] {
  const commonWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall']
  
  const sentences = text.split(/[.!?]+/)
  const phrases: string[] = []
  
  sentences.forEach(sentence => {
    const words = sentence.trim().split(/\s+/)
    if (words.length >= 3) {
      for (let i = 0; i <= words.length - 3; i++) {
        const phrase = words.slice(i, i + Math.min(8, words.length - i)).join(' ')
        if (phrase.length <= maxLength && phrase.length >= 10) {
          const firstWord = words[i].toLowerCase().replace(/[^a-z]/g, '')
          const lastWord = words[i + Math.min(7, words.length - i - 1)].toLowerCase().replace(/[^a-z]/g, '')
          
          if (!commonWords.includes(firstWord) || !commonWords.includes(lastWord)) {
            phrases.push(phrase.trim())
          }
        }
      }
    }
  })
  
  return [...new Set(phrases)].sort((a, b) => b.length - a.length).slice(0, 3)
}

export function findBestHighlightText(referenceText: string): string {
  const phrases = extractKeyPhrasesForHighlight(referenceText, 40)
  return phrases[0] || referenceText.substring(0, 30)
}