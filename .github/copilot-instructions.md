# Coding Agent Instructions

## Code Standards

### Style and Structure
- Write minimalistic code that solves the problem directly
- Prefer functional programming patterns over object-oriented approaches
- Favor composition over inheritance when objects are necessary
- Keep functions small and focused on a single responsibility

### Error Handling
- Fail fast: error out eagerly rather than attempting to catch and recover
- Let errors propagate naturally instead of swallowing exceptions
- Use assertions for invariants and preconditions
- Validate input at boundaries, then trust it internally

### Code Clarity
- Prioritize fewer lines of code over verbose implementations
- Avoid comments for self-explanatory code
- Use descriptive variable and function names that eliminate the need for comments
- Add precise, detailed comments only for complex algorithms or non-obvious logic
- Comment on "why" not "what" when explanation is needed

## Documentation

### General Documentation and Formatting
- Do not use Emojis

### Code documentation
- Write clear and concise docstrings only where required to understand the function or class