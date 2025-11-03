export default [
  {
    ignores: ['dist/**', 'node_modules/**', 'venv/**', '__pycache__/**', 'docs/**'],
  },
  {
    files: ['js/**/*.js', 'scripts/**/*.js', 'test/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        console: 'readonly',
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        requestIdleCallback: 'readonly',
        cancelIdleCallback: 'readonly',
        performance: 'readonly',
        scheduler: 'readonly',
        document: 'readonly',
        URL: 'readonly',
        global: 'writable',
        NOISEMAKER_PRESETS_DSL: 'readonly',
        glData: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': ['error', { 
        argsIgnorePattern: '^_|^shape$|^time$|^speed$', 
        varsIgnorePattern: '^_|^h$|^w$|^c$',
        destructuredArrayIgnorePattern: '^_',
        caughtErrors: 'none'
      }],
      'no-undef': 'error',
    },
  },
];
