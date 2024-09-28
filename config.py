from guardrails import Guard
from guardrails.hub import GibberishText
guard = Guard()
guard.name = 'gibberish_guard'
print("GUARD PARAMETERS UNFILLED! UPDATE THIS FILE!")  # TODO: Remove this when parameters are filled.
guard.use(GibberishText())  # TODO: Add parameters.