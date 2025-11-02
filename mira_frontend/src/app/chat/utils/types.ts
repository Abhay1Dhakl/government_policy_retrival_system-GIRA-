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
    id: 'grds',
    label: 'GRDs',
    enabled: true,
    toolName: 'grd'
  },
  {
    id: 'lrds',
    label: 'LRDs',
    enabled: true,
    toolName: 'lrd'
  },
  {
    id: 'escalations',
    label: 'Escalations, PIs, Leaflets',
    enabled: true,
    toolName: 'pis'
  },
  {
    id: 'past-cases',
    label: 'Past Cases',
    enabled: true,
    toolName: 'past_cases'
  },
  {
    id: 'online-databases',
    label: 'Online Databases-Embase,Pubmed',
    enabled: true,
    toolName: 'online_db'
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