export interface DocumentTypeOption {
  id: string;
  label: string;
  enabled: boolean;
  toolName: string;
}

export interface LLMOption {
  id: string;
  label: string;
  selected: boolean;
  apiName: string;
}

export const documentTypeOptions: DocumentTypeOption[] = [
  {
    id: 'general',
    label: 'All Documents',
    enabled: true,
    toolName: 'general_search'
  },
  {
    id: 'constitution',
    label: 'Constitution',
    enabled: true,
    toolName: 'constitution'
  },
  {
    id: 'acts',
    label: 'Acts & Legislation',
    enabled: true,
    toolName: 'acts'
  },
  {
    id: 'regulations',
    label: 'Regulations & Bylaws',
    enabled: true,
    toolName: 'regulations'
  },
  {
    id: 'past-cases',
    label: 'Past Cases',
    enabled: true,
    toolName: 'past_cases'
  },
];

export const llmOptions: LLMOption[] = [
  {
    id: 'chatgpt',
    label: 'ChatGPT',
    selected: true,
    apiName: 'openai'
  },
  {
    id: 'claude',
    label: 'Claude',
    selected: false,
    apiName: 'claude'
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    selected: false,
    apiName: 'anthropic'
  },
  {
    id: 'gemini',
    label: 'Gemini',
    selected: false,
    apiName: 'gemini'
  },
  {
    id: 'llama',
    label: 'Llama',
    selected: false,
    apiName: 'llama'
  },
  {
    id: 'deepseek',
    label: 'Deepseek',
    selected: false,
    apiName: 'deepseek'
  },
  {
    id: 'grok',
    label: 'Grok',
    selected: false,
    apiName: 'grok'
  },
];