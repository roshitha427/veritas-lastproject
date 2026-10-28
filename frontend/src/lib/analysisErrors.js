const legacyCaptionRule = /(?:at least\s+10\s+characters?|10\s+characters?.{0,100}(?:claim|caption))/i;

export function analysisErrorMessage(error) {
  const message = typeof error === 'string' ? error : error?.message;
  if (message && legacyCaptionRule.test(message)) {
    return 'The analysis server needs to be restarted before this upload can be checked. Please try again after it reloads.';
  }
  return message || 'The media could not be analyzed. Please try again.';
}
