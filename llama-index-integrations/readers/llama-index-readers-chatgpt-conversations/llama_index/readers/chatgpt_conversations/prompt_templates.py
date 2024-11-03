default_title_single_template = """\
You are a title generator. Generate a title that summarizes all of \
the unique entities, titles or themes found in the context.

**Correct Example:**
Context: Suffice to say, we can safely ignore all the rebuttals from the plaintiff given that there is no hard evidence provided. However, if you have evidence to the contrary, you stand to completely expunge that allegation altogether.
Title: Overturning Allegations due to Lack of Evidence

**Wrong Format:**
Context: Suffice to say, we can safely ignore all the rebuttals from the plaintiff given that there is no hard evidence provided. However, if you have evidence to the contrary, you stand to completely expunge that allegation altogether.
Title: Based on the content provided, it appears that the comprehensive title for this document could be: "Overturning Allegations due to Lack of Evidence"

!Do not add any additional words or comments, simply respond with the title for the given context!

Context: {context_str}.
Title: """

default_title_combine_template = """\
You are a title summarizer. Based on the above candidate titles and content, \
what is the comprehensive title for this document?

**Correct Example:**
Context: "Crafting Defence Through Absence", "Overturning Allegations due to Lack of Evidence", "When Absence Speaks in Legal Discourse"
Title: Defence in Absence - The Strategic Weight of Unproven Claims

**Wrong Format:**
Context: Suffice to say, we can safely ignore all the rebuttals from the plaintiff given that there is no hard evidence provided. However, if you have evidence to the contrary, you stand to completely expunge that allegation altogether.
Title: Based on the content provided, it appears that the comprehensive title for this document could be: "Defence in Absence - The Strategic Weight of Unproven Claims"

!Do not add any additional words or comments, simply respond with the title for the given context!

Context: {context_str}.
Title: """
