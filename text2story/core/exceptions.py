"""
	text2story.core.exceptions

	All defined exceptions raised by the package.
"""

class InvalidLanguage(Exception):
	"""
	Raised if the user specified an invalid/unsupported language.
	"""
	def __init__(self, lang):
		description = ('Invalid language :' + lang)
		
		super().__init__(description)

class InvalidTool(Exception):
	"""
	Raised if the user specified an invalid/unsupported tool.
	"""

	def __init__(self, tool):
		description = ('Invalid tool: ' + tool)
		
		super().__init__(description)
