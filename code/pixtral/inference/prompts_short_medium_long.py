# EMOTIONS WITH CORRELATIONS SHORT
emotions_correlations_short = """You are an expert in emotion classification. Classify the emotions expressed in a Twitter post written in Italian, consisting of an image and a caption-like text. Choose 1-2 emotions from: Love, Anticipation, Disgust, Trust, Joy, Fear, Anger, Surprise, Sadness, Neutral. Consider the following relationships: Disgust correlates with Sarcasm. Anger correlates with Offensivity. Surprise correlates with Content. Trust correlates negatively with Irony and Divergent.

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITH CORRELATIONS MEDIUM
emotions_correlations_medium = """You are an expert in emotion classification. Classify the emotions expressed in a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact to convey emotion, as they do in real Twitter posts. Choose 1-2 emotions from: Love, Anticipation, Disgust, Trust, Joy, Fear, Anger, Surprise, Sadness, Neutral.

The posts are from two Italian TV shows: Pechino Express (a travel-based reality show where contestants face challenges and cultural exchanges) and Propaganda Live (a satirical talk show mixing politics, field reports, debates, music, and guests).

Also consider the following relationships: Disgust correlates with Sarcasm, Content and Offensivness, but is negatively correlated with Container. Trust correlates with Container and negatively with Irony and Divergent. Anger correlates with Offensivity. Surprise correlates with Content and Divergent, but is negatively correlated with Container. Sadness correlates with Content focus. Anticipation correlates with Contentainer.

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITH CORRELATIONS LONG - VERSION 1
emotions_correlations_long_v1 = """You are an expert in emotion classification. Classify the emotions expressed in a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact to convey emotion, as they do in real Twitter posts.

Use only the following set of emotions:
- Love (Affection or appreciation toward someone)
- Anticipation (Expectation of a future event)
- Disgust (Aversion or repulsion)
- Trust (Confidence, admiration, or respect)
- Joy (Positive emotions such as happiness or amusement)
- Fear (Response to threats or danger)
- Anger (Irritation, frustration, or indignation)
- Surprise (Reaction to the unexpected)
- Sadness (Sorrow, grief, or unhappiness)
- Neutral (No strong emotional expression)

Classify the post using only 1 or 2 emotions from this list. If no specific emotion is clearly expressed, select Neutral. Pay special attention to irony: when it is present, take it into account during classification, as it often shifts the intended emotional tone of the text.

The posts are from two Italian TV shows:
- Pechino Express: A travel-based reality show where contestants face challenges and cultural exchanges with little money.
- Propaganda Live: A satirical talk show mixing politics, field reports, debates, music, and guests.

Also consider the following relationships between emotional expression and other annotation levels:
- Trust correlates with Container and Parallel relationships, and negatively with Irony, Divergent and Content.
- Surprise correlates with Content, Divergent relationships, and Irony and negatively with Container.
- Sadness correlates with Content and Additive relationships, but negatively with Container.
- Anticipation correlates with Content and negatively with Container.
- Disgust correlates with Sarcasm, Divergent relationships, Offensiveness and Content and negatively with Parallel relationships and Container
- Anger correlates with Offensiveness and Sarcasm
- Joy correlates with Irony

The sentences to analyze are always preceded by additional information from other annotation layers.

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITH CORRELATIONS LONG - VERSION 2
emotions_correlations_long_v2 = """As an emotion classification specialist, your task involves analyzing Italian Twitter posts that combine visual and textual elements. Each post contains an image paired with caption-like text. Your goal is to examine how these multimodal components work together to express emotions, mirroring authentic Twitter interactions.

Select 1-2 emotions exclusively from this predefined set:
- Love (expressing affection or appreciation toward someone)
- Anticipation (conveying expectation of future events)
- Disgust (showing aversion or repulsion)
- Trust (demonstrating confidence, admiration, or respect)
- Joy (reflecting positive emotions like happiness or amusement)
- Fear (indicating response to threats or danger)
- Anger (displaying irritation, frustration, or indignation)
- Surprise (revealing reaction to unexpected events)
- Sadness (expressing sorrow, grief, or unhappiness)
- Neutral (indicating absence of strong emotional expression)

When no distinct emotion emerges clearly, default to Neutral. Irony requires particular attention—its presence frequently alters the intended emotional meaning of the content.

Context: These posts originate from two Italian television programs:
- Pechino Express: A reality travel show featuring contestants who navigate challenges and cultural interactions with limited financial resources.
- Propaganda Live: A satirical talk show that blends political commentary, field reporting, debates, musical performances, and guest appearances.

Additional information from other annotation layers always precedes the content to be analyzed.

Key correlational patterns to consider:
- Trust: positively linked with Container and Parallel relationships, negatively associated with Irony, Divergent relationships, and Content focus
- Surprise: connected to Content focus, Divergent relationships, and Irony; negatively associated with Container relationships
- Sadness: associated with Content focus and Additive relationships; negatively associated with Container relationships
- Anticipation: linked to Content focus; negatively associated with Container relationships
- Disgust: correlates with Sarcasm, Offensiveness, Divergent relationships, and Content focus; negatively correlates with Parallel and Container relationships
- Anger: associated with Offensiveness and Sarcasm
- Joy: connected to Irony

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITH CORRELATIONS LONG - VERSION 3
emotions_correlations_long_v3 = """Your expertise in emotion classification is needed to evaluate Italian Twitter posts featuring both visual and textual content. Each post presents an image accompanied by text that functions as a caption or comment. Analyze how these elements combine to communicate emotional meaning, reflecting real Twitter posting behavior.

Choose strictly 1 or 2 emotions from the following categories:
- Love: Demonstrates affection or appreciation directed toward someone
- Anticipation: Indicates expectation or looking forward to future events
- Disgust: Reveals aversion, repulsion, or strong dislike
- Trust: Shows confidence, admiration, or respectful regard
- Joy: Encompasses positive emotional states including happiness and amusement
- Fear: Represents responses to perceived threats or dangerous situations
- Anger: Encompasses irritation, frustration, or feelings of indignation
- Surprise: Captures reactions to unexpected or unanticipated events
- Sadness: Includes sorrow, grief, or general unhappiness
- Neutral: Applied when no pronounced emotional expression is evident

If emotional content appears ambiguous or unclear, select Neutral. Be particularly mindful of ironic content, which commonly transforms the apparent emotional tone of the message.

Source material comes from these two Italian TV shows:
- Pechino Express: Reality-based travel program where participants encounter challenges and engage in cultural exchanges while operating with minimal funds
- Propaganda Live: Satirical talk show format incorporating political analysis, field journalism, debates, musical segments, and guest interviews

Sentences for analysis are consistently prefaced with supplementary information from additional annotation categories.

Relevant correlational relationships include:
- Trust: positively correlates with Container and Parallel relationships; exhibits negative correlation with Irony, Divergent relationships, and Content focus
- Surprise: demonstrates positive correlations with Content focus, Divergent relationships, and Irony; shows negative correlation with Container relationships
- Sadness: shows positive correlation with both Content focus and Additive relationships; exhibits negative correlation with Container relationships
- Anticipation: positively correlates with Content focus; negatively correlates with Container relationships
- Disgust: correlates positively with Sarcasm, Offensiveness, Divergent relationships, and Content focus; shows negative correlation with Parallel and Container relationships
- Anger: exhibits positive correlations with both Offensiveness and Sarcasm
- Joy: demonstrates positive correlation with Irony

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITHOUT CORRELATIONS SHORT
emotions_no_correlations_short = """You are an expert in emotion classification. Classify the emotions expressed in a Twitter post written in Italian, consisting of an image and a caption-like text. Choose 1-2 emotions from: Love, Anticipation, Disgust, Trust, Joy, Fear, Anger, Surprise, Sadness, Neutral.

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITHOUT CORRELATIONS MEDIUM
emotions_no_correlations_medium = """You are an expert in emotion classification. Classify the emotions expressed in a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact to convey emotion, as they do in real Twitter posts. Choose 1-2 emotions from: Love, Anticipation, Disgust, Trust, Joy, Fear, Anger, Surprise, Sadness, Neutral.

The posts are from two Italian TV shows: Pechino Express (a travel-based reality show where contestants face challenges and cultural exchanges) and Propaganda Live (a satirical talk show mixing politics, field reports, debates, music, and guests).

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITHOUT CORRELATIONS LONG - VERSION 1
emotions_no_correlations_long_v1 = """You are an expert in emotion classification. Classify the emotions expressed in a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact to convey emotion, as they do in real Twitter posts.

Use only the following set of emotions:
- Love (Affection or appreciation toward someone)
- Anticipation (Expectation of a future event)
- Disgust (Aversion or repulsion)
- Trust (Confidence, admiration, or respect)
- Joy (Positive emotions such as happiness or amusement)
- Fear (Response to threats or danger)
- Anger (Irritation, frustration, or indignation)
- Surprise (Reaction to the unexpected)
- Sadness (Sorrow, grief, or unhappiness)
- Neutral (No strong emotional expression)

Classify the post using only 1 or 2 emotions from this list. If no specific emotion is clearly expressed, select Neutral. Pay special attention to irony: when it is present, take it into account during classification, as it often shifts the intended emotional tone of the text.

The posts are from two Italian TV shows:
- Pechino Express: A travel-based reality show where contestants face challenges and cultural exchanges with little money.
- Propaganda Live: A satirical talk show mixing politics, field reports, debates, music, and guests.

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITHOUT CORRELATIONS LONG - VERSION 2
emotions_no_correlations_long_v2 = """As a specialist in emotion classification, analyze Italian Twitter posts that combine visual and textual elements. Each post features an image with accompanying caption-like text. Your objective is to examine how these multimodal components interact to express emotions, reflecting authentic Twitter usage patterns.

Select 1-2 emotions exclusively from this defined set:
- Love (expressing affection or appreciation toward someone)
- Anticipation (conveying expectation of future events)
- Disgust (showing aversion or repulsion)
- Trust (demonstrating confidence, admiration, or respect)
- Joy (reflecting positive emotions like happiness or amusement)
- Fear (indicating response to threats or danger)
- Anger (displaying irritation, frustration, or indignation)
- Surprise (revealing reaction to unexpected events)
- Sadness (expressing sorrow, grief, or unhappiness)
- Neutral (indicating absence of strong emotional expression)

When emotional content remains unclear or ambiguous, choose Neutral. Give particular consideration to ironic elements, which frequently modify the apparent emotional meaning of the content.

These posts originate from two Italian television programs:
- Pechino Express: A reality travel show where participants navigate challenges and cultural interactions with limited financial resources.
- Propaganda Live: A satirical talk show that combines political commentary, field reporting, debates, musical performances, and guest appearances.

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# EMOTIONS WITHOUT CORRELATIONS LONG - VERSION 3
emotions_no_correlations_long_v3 = """Your expertise in emotion classification is required to evaluate Italian Twitter posts that incorporate both visual and textual components. Each post presents an image paired with text functioning as a caption or commentary. Determine how these elements work together to communicate emotional meaning, mirroring genuine Twitter interactions.

Choose strictly 1 or 2 emotions from these categories:
- Love: Demonstrates affection or appreciation directed toward someone
- Anticipation: Indicates expectation or anticipation of future events
- Disgust: Reveals aversion, repulsion, or strong dislike
- Trust: Shows confidence, admiration, or respectful regard
- Joy: Encompasses positive emotional states including happiness and amusement
- Fear: Represents responses to perceived threats or dangerous situations
- Anger: Encompasses irritation, frustration, or feelings of indignation
- Surprise: Captures reactions to unexpected or unanticipated events
- Sadness: Includes sorrow, grief, or general unhappiness
- Neutral: Applied when no pronounced emotional expression is evident

If emotional expression appears indistinct or unclear, select Neutral. Be especially attentive to ironic content, which commonly alters the perceived emotional tone of the message.

Source content derives from these two Italian TV shows:
- Pechino Express: Reality-based travel program where contestants encounter challenges and engage in cultural exchanges while operating with minimal funds
- Propaganda Live: Satirical talk show format incorporating political analysis, field journalism, debates, musical segments, and guest interviews

Your response must be without any additional text or explanations, and formatted as follows:
Emotions detected: [Emotion(s)]"""

# RELATIONS WITH CORRELATIONS SHORT
relations_correlations_short = """You are an expert in multimodal classification. Classify the relationship between an Italian Twitter post's text and image. Choose only one of the following categories: Additive (the text and image enrich each other), Parallel (the text and image express the same idea), Divergent (the text and image convey contrasting meanings, often producing irony or humor). Helpful correlations: Divergent correlates with Irony and Content focus. Additive correlates with Container.

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITH CORRELATIONS MEDIUM
relations_correlations_medium = """You are an expert in multimodal classification. Classify the relationship between an Italian Twitter post's text and image. They appear together like a real post: the text is a caption or comment on the image. Choose only one of the following categories: Additive (the text and image enrich each other), Parallel (the text and image express the same idea), Divergent (the text and image convey contrasting meanings, often producing irony or humor).

Posts refer to two Italian TV shows: Pechino Express (a travel reality show with challenges and cultural exchange) and Propaganda Live (a satirical talk show mixing politics, reports, music, and guest commentary).

Helpful correlations: Divergent correlates with Irony, Content focus and Surprise, and negatively with Trust and Container. Additive correlates with Sadness and Container and negatively with Irony. Parallel correlates with Trust.

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITH CORRELATIONS LONG - VERSION 1
relations_correlations_long_v1 = """You are an expert in multimodal classification. Classify the relationship between an Italian Twitter post's text and image. They appear together like a real post: the text is a caption or comment on the image.

Choose only one of the following categories:
- Additive: the text and image enrich each other to form a deeper or more amplified meaning
- Parallel: the text and image express the same idea or sentiment independently
- Divergent: the text and image convey contrasting or conflicting meanings, often producing irony, humor, or nonsense

Posts refer to two Italian TV shows:
- Pechino Express: a travel reality show with challenges and cultural exchange
- Propaganda Live: a satirical talk show mixing politics, reports, music, and guest commentary

Guidelines:
- If the text and image clearly express the same idea: Parallel
- If they combine into a richer meaning: Additive
- If they conflict in tone, message, or intent: Divergent

Posts also include annotations such as emotions, irony, offensiveness, and target (Content vs. Container).

Helpful correlations:
- Divergent correlates with Irony, Sarcasm, Content focus, Disgust, Surprise, and negatively with Trust and Container
- Additive correlates with Trust, Anticipation, Sadness, and Container focus; negatively correlates with Content and Irony
- Parallel correlates with Trust and Container

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITH CORRELATIONS LONG - VERSION 2
relations_correlations_long_v2 = """As a multimodal classification specialist, analyze the relationship between textual and visual elements in Italian Twitter posts. These posts function as authentic social media content where text serves as a caption or commentary for the accompanying image.

Select exactly one category from these options:
- Additive: textual and visual components enhance each other, creating deeper or amplified meaning
- Parallel: text and image independently express identical ideas or sentiments
- Divergent: text and image present contrasting or conflicting meanings, frequently generating irony, humor, or absurdity

Content originates from two Italian television programs:
- Pechino Express: reality-based travel show featuring challenges and cultural exchange experiences
- Propaganda Live: satirical talk show format combining political commentary, field reports, musical elements, and guest discussions

Classification guidelines:
- When text and image convey identical concepts: select Parallel
- When they merge to create enriched meaning: select Additive
- When they clash in tone, message, or purpose: select Divergent

Additional annotations accompany these posts, including emotional markers, irony indicators, offensiveness ratings, and target classifications (Content vs. Container).

Relevant correlational patterns:
- Divergent: positively correlates with Irony, Sarcasm, Content focus, Disgust, Surprise; negatively correlates with Trust and Container
- Additive: positively correlates with Trust, Anticipation, Sadness, and Container focus; negatively correlates with Content and Irony
- Parallel: positively correlates with Trust and Container

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITH CORRELATIONS LONG - VERSION 3
relations_correlations_long_v3 = """Your expertise in multimodal classification is needed to determine the relationship between text and images in Italian Twitter posts. These posts mirror real social media interactions where textual content functions as captions or comments accompanying visual elements.

Choose one classification from these three categories:
- Additive: text and image work together synergistically, producing enriched or intensified meaning beyond individual components
- Parallel: textual and visual elements independently communicate the same concept or emotional sentiment
- Divergent: text and image present opposing or contradictory meanings, typically resulting in ironic, humorous, or nonsensical effects

Source material derives from two Italian TV shows:
- Pechino Express: travel-focused reality program incorporating challenges and cross-cultural interactions
- Propaganda Live: satirical talk show blending political analysis, investigative reporting, musical performances, and guest commentary

Decision framework:
- Identical expression across modalities: choose Parallel
- Synergistic combination creating enhanced meaning: choose Additive
- Contradictory tone, messaging, or intent: choose Divergent

Posts include supplementary annotations covering emotional content, ironic elements, offensive material, and focus targets (Content vs. Container).

Key correlational relationships:
- Divergent: shows positive correlations with Irony, Content focus, Disgust, Surprise; exhibits negative correlation with Trust and Container
- Additive: demonstrates positive correlations with Trust, Anticipation, Sadness, and Container focus; exhibits negative correlation with Content and Irony
- Parallel: correlates positively Trust and Container

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITHOUT CORRELATIONS SHORT
relations_no_correlations_short = """You are an expert in multimodal classification. Classify the relationship between an Italian Twitter post's text and image. Choose only one of the following categories: Additive (the text and image enrich each other), Parallel (the text and image express the same idea), Divergent (the text and image convey contrasting meanings, often producing irony or humor).

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITHOUT CORRELATIONS MEDIUM
relations_no_correlations_medium = """You are an expert in multimodal classification. Classify the relationship between an Italian Twitter post's text and image. They appear together like a real post: the text is a caption or comment on the image. Choose only one of the following categories: Additive (the text and image enrich each other), Parallel (the text and image express the same idea), Divergent (the text and image convey contrasting meanings, often producing irony or humor).

Posts refer to two Italian TV shows: Pechino Express (a travel reality show with challenges and cultural exchange) and Propaganda Live (a satirical talk show mixing politics, reports, music, and guest commentary).

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITHOUT CORRELATIONS LONG - VERSION 1
relations_no_correlations_long_v1 = """You are an expert in multimodal classification. Classify the relationship between an Italian Twitter post's text and image. They appear together like a real post: the text is a caption or comment on the image.

Choose only one of the following categories:
- Additive: the text and image enrich each other to form a deeper or more amplified meaning
- Parallel: the text and image express the same idea or sentiment independently
- Divergent: the text and image convey contrasting or conflicting meanings, often producing irony, humor, or nonsense

Posts refer to two Italian TV shows:
- Pechino Express: a travel reality show with challenges and cultural exchange
- Propaganda Live: a satirical talk show mixing politics, reports, music, and guest commentary

Guidelines:
- If the text and image clearly express the same idea: Parallel
- If they combine into a richer meaning: Additive
- If they conflict in tone, message, or intent: Divergent

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITHOUT CORRELATIONS LONG - VERSION 2
relations_no_correlations_long_v2 = """As a specialist in multimodal classification, determine the relationship between textual and visual components in Italian Twitter posts. These posts function as authentic social media content where text operates as a caption or commentary for the associated image.

Select exactly one category from these options:
- Additive: textual and visual elements enhance each other, creating deeper or amplified meaning
- Parallel: text and image independently express identical ideas or sentiments
- Divergent: text and image present contrasting or conflicting meanings, frequently generating irony, humor, or absurdity

Content originates from two Italian television programs:
- Pechino Express: reality-based travel show featuring challenges and cultural exchange experiences
- Propaganda Live: satirical talk show format combining political commentary, field reports, musical elements, and guest discussions

Classification guidelines:
- When text and image convey identical concepts: select Parallel
- When they merge to create enriched meaning: select Additive
- When they clash in tone, message, or purpose: select Divergent

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""

# RELATIONS WITHOUT CORRELATIONS LONG - VERSION 3
relations_no_correlations_long_v3 = """Your expertise in multimodal classification is required to analyze the relationship between text and images in Italian Twitter posts. These posts reflect genuine social media interactions where textual content serves as captions or comments for accompanying visual elements.

Choose one classification from these three categories:
- Additive: text and image work together synergistically, producing enriched or intensified meaning beyond individual components
- Parallel: textual and visual elements independently communicate the same concept or emotional sentiment
- Divergent: text and image present opposing or contradictory meanings, typically resulting in ironic, humorous, or nonsensical effects

Source material derives from two Italian TV shows:
- Pechino Express: travel-focused reality program incorporating challenges and cross-cultural interactions
- Propaganda Live: satirical talk show blending political analysis, investigative reporting, musical performances, and guest commentary

Decision framework:
- Identical expression across modalities: choose Parallel
- Synergistic combination creating enhanced meaning: choose Additive
- Contradictory tone, messaging, or intent: choose Divergent

Your response must be without any additional text or explanations, and formatted as follows:
Relationship detected: [Additive/Divergent/Parallel]"""



# IRONY WITH CORRELATIONS SHORT
irony_correlations_short = """You are an expert in irony and sarcasm classification. Classify a Twitter post written in Italian, consisting of an image and a caption-like text. Choose only 1 category from: irony, sarcasm, or none (unironic).

Consider the following relationships: irony correlates positively with Divergent and Content. irony correlates negatively with Container, Additive, and Trust. sarcasm correlates with Disgust. none correlates positively with Container, Trust, and Additive. none correlates negatively with Divergent and Content.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# IRONY WITHOUT CORRELATIONS SHORT
irony_no_correlations_short = """You are an expert in irony and sarcasm classification. Classify a Twitter post written in Italian, consisting of an image and a caption-like text. Choose only 1 category from: irony, sarcasm, or none (unironic).

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# --- MEDIUM PROMPTS ---

# IRONY WITH CORRELATIONS MEDIUM
irony_correlations_medium = """You are an expert in irony and sarcasm classification. Classify a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact to convey meaning, as they do in real Twitter posts. Choose only 1 category from: irony, sarcasm, or none (unironic).

The posts are from two Italian TV shows: Pechino Express (a travel-based reality show where contestants face challenges and cultural exchanges) and Propaganda Live (a satirical talk show mixing politics, field reports, debates, music, and guests).

Also consider the following relationships: irony correlates positively with Divergent and Content. irony correlates negatively with Container, Additive, and Trust. sarcasm correlates with Disgust. none correlates positively with Container, Trust, and Additive. none correlates negatively with Divergent and Content.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# IRONY WITHOUT CORRELATIONS MEDIUM
irony_no_correlations_medium = """You are an expert in irony and sarcasm classification. Classify a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact to convey meaning, as they do in real Twitter posts. Choose only 1 category from: irony, sarcasm, or none (unironic).

The posts are from two Italian TV shows: Pechino Express (a travel-based reality show where contestants face challenges and cultural exchanges) and Propaganda Live (a satirical talk show mixing politics, field reports, debates, music, and guests).

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# --- LONG PROMPTS (WITH CORRELATIONS) ---

# IRONY WITH CORRELATIONS LONG - VERSION 1
irony_correlations_long_v1 = """You are an expert in irony and sarcasm classification. Classify a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact, as they do in real Twitter posts.

Use only the following set of categories:
- irony: Expressing meaning using language that normally signifies the opposite.
- sarcasm: A sharp, aggressive irony intended to hurt or criticize a specific target.
- none: No irony or sarcasm is present; the post is literal and unironic.

Classify the post using only 1 category from this list. If no irony or sarcasm is present, select none.

The posts are from two Italian TV shows:
- Pechino Express: A travel-based reality show where contestants face challenges and cultural exchanges with little money.
- Propaganda Live: A satirical talk show mixing politics, field reports, debates, music, and guests.

Also consider the following relationships between the classification and other annotation levels:
- irony correlates with Divergent, Content, Surprise, and Joy, and negatively with Container, Additive, Trust, and Parallel.
- sarcasm correlates with Disgust, Offensiveness, and Anger.
- none correlates with Container, Trust, Additive, and Parallel, and negatively with Divergent, Content, Disgust, and Surprise.

The sentences to analyze are always preceded by additional information from other annotation layers.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# IRONY WITH CORRELATIONS LONG - VERSION 2
irony_correlations_long_v2 = """As an irony and sarcasm classification specialist, your task involves analyzing Italian Twitter posts that combine visual and textual elements. Each post contains an image paired with caption-like text. Your goal is to examine how these multimodal components work together, mirroring authentic Twitter interactions.

Select 1 category exclusively from this predefined set:
- irony: A form of expression where the intended meaning is the opposite of the literal meaning.
- sarcasm: Biting, cutting irony with a clear intent to mock, wound, or show contempt.
- none: The content is straightforward and does not use ironic or sarcastic devices.

When no irony or sarcasm emerges clearly, default to none.

Context: These posts originate from two Italian television programs:
- Pechino Express: A reality travel show featuring contestants who navigate challenges and cultural interactions with limited financial resources.
- Propaganda Live: A satirical talk show that blends political commentary, field reporting, debates, musical performances, and guest appearances.

Additional information from other annotation layers always precedes the content to be analyzed.

Key correlational patterns to consider:
- irony: positively linked with Divergent, Content, Surprise, and Joy; negatively associated with Container, Additive, Trust, and Parallel.
- sarcasm: associated with Disgust, Offensiveness, and Anger.
- none: positively associated with Container, Trust, Additive, and Parallel; negatively linked with Divergent, Content, Disgust, and Surprise.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# IRONY WITH CORRELATIONS LONG - VERSION 3
irony_correlations_long_v3 = """Your expertise in irony and sarcasm classification is needed to evaluate Italian Twitter posts featuring both visual and textual content. Each post presents an image accompanied by text that functions as a caption or comment. Analyze how these elements combine to communicate meaning, reflecting real Twitter posting behavior.

Choose strictly 1 category from the following:
- irony: The use of words to convey a meaning that is the opposite of its literal sense.
- sarcasm: Sharp, bitter irony, often driven by anger or negativity towards a clear target.
- none: Represents literal, unironic, and direct communication.

If ironic or sarcastic content appears ambiguous or unclear, select none.

Source material comes from these two Italian TV shows:
- Pechino Express: Reality-based travel program where participants encounter challenges and engage in cultural exchanges while operating with minimal funds
- Propaganda Live: Satirical talk show format incorporating political analysis, field journalism, debates, musical segments, and guest interviews

Sentences for analysis are consistently prefaced with supplementary information from additional annotation categories.

Relevant correlational relationships include:
- irony: positively correlates with Divergent, Content, Surprise, and Joy; exhibits negative correlation with Container, Additive, Trust, and Parallel.
- sarcasm: correlates positively with Disgust, Offensiveness, and Anger.
- none: shows positive correlations with Container, Trust, Additive, and Parallel; negatively correlates with Divergent, Content, Disgust, and Surprise.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# --- LONG PROMPTS (WITHOUT CORRELATIONS) ---

# IRONY WITHOUT CORRELATIONS SHORT - VERSION 1
irony_no_correlations_long_v1 = """You are an expert in irony and sarcasm classification. Classify a Twitter post written in Italian, consisting of an image and a caption-like text. The task is to analyze both the image and the accompanying text, considering how they interact, as they do in real Twitter posts.

Use only the following set of categories:
- irony: Expressing meaning using language that normally signifies the opposite.
- sarcasm: A sharp, aggressive irony intended to hurt or criticize a specific target.
- none: No irony or sarcasm is present; the post is literal and unironic.

Classify the post using only 1 category from this list. If no irony or sarcasm is present, select none.

The posts are from two Italian TV shows:
- Pechino Express: A travel-based reality show where contestants face challenges and cultural exchanges with little money.
- Propaganda Live: A satirical talk show mixing politics, field reports, debates, music, and guests.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# IRONY WITHOUT CORRELATIONS LONG - VERSION 2
irony_no_correlations_long_v2 = """As a specialist in irony and sarcasm classification, analyze Italian Twitter posts that combine visual and textual elements. Each post features an image with accompanying caption-like text. Your objective is to examine how these multimodal components interact, reflecting authentic Twitter usage patterns.

Select 1 category exclusively from this defined set:
- irony: A form of expression where the intended meaning is the opposite of the literal meaning.
- sarcasm: Biting, cutting irony with a clear intent to mock, wound, or show contempt.
- none: The content is straightforward and does not use ironic or sarcastic devices.

When ironic or sarcastic content remains unclear or ambiguous, choose none.

These posts originate from two Italian television programs:
- Pechino Express: A reality travel show where participants navigate challenges and cultural interactions with limited financial resources.
- Propaganda Live: A satirical talk show that combines political commentary, field reporting, debates, musical performances, and guest appearances.

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# IRONY WITHOUT CORRELATIONS LONG - VERSION 3
irony_no_correlations_long_v3 = """Your expertise in irony and sarcasm classification is required to evaluate Italian Twitter posts that incorporate both visual and textual components. Each post presents an image paired with text functioning as a caption or commentary. Determine how these elements work together, mirroring genuine Twitter interactions.

Choose strictly 1 category from these:
- irony: The use of words to convey a meaning that is the opposite of its literal sense.
- sarcasm: Sharp, bitter irony, often driven by anger or negativity towards a clear target.
- none: Represents literal, unironic, and direct communication.

If ironic or sarcastic expression appears indistinct or unclear, select none.

Source content derives from these two Italian TV shows:
- Pechino Express: Reality-based travel program where contestants encounter challenges and engage in cultural exchanges while operating with minimal funds
- Propaganda Live: Satirical talk show format incorporating political analysis, field journalism, debates, musical segments, and guest interviews

Your response must be without any additional text or explanations, and formatted as follows:
Label detected: [Irony/Sarcasm/None]"""

# Dictionary for easy access
prompts = {
    "emotions_correlations_short": emotions_correlations_short,
    "emotions_correlations_medium": emotions_correlations_medium,
    "emotions_correlations_long_v1": emotions_correlations_long_v1,
    "emotions_correlations_long_v2": emotions_correlations_long_v2,
    "emotions_correlations_long_v3": emotions_correlations_long_v3,
    "emotions_no_correlations_short": emotions_no_correlations_short,
    "emotions_no_correlations_medium": emotions_no_correlations_medium,
    "emotions_no_correlations_long_v1": emotions_no_correlations_long_v1,
    "emotions_no_correlations_long_v2": emotions_no_correlations_long_v2,
    "emotions_no_correlations_long_v3": emotions_no_correlations_long_v3,
    "relations_correlations_short": relations_correlations_short,
    "relations_correlations_medium": relations_correlations_medium,
    "relations_correlations_long_v1": relations_correlations_long_v1,
    "relations_correlations_long_v2": relations_correlations_long_v2,
    "relations_correlations_long_v3": relations_correlations_long_v3,
    "relations_no_correlations_short": relations_no_correlations_short,
    "relations_no_correlations_medium": relations_no_correlations_medium,
    "relations_no_correlations_long_v1": relations_no_correlations_long_v1,
    "relations_no_correlations_long_v2": relations_no_correlations_long_v2,
    "relations_no_correlations_long_v3": relations_no_correlations_long_v3,
    "irony_correlations_short": irony_correlations_short,
    "irony_correlations_medium": irony_correlations_medium,
    "irony_correlations_long_v1": irony_correlations_long_v1,
    "irony_correlations_long_v2": irony_correlations_long_v2,
    "irony_correlations_long_v3": irony_correlations_long_v3,
    "irony_no_correlations_short": irony_no_correlations_short,
    "irony_no_correlations_medium": irony_no_correlations_medium,
    "irony_no_correlations_long_v1": irony_no_correlations_long_v1,
    "irony_no_correlations_long_v2": irony_no_correlations_long_v2,
    "irony_no_correlations_long_v3": irony_no_correlations_long_v3
}

# Function to get a specific prompt
def get_prompt(prompt_name):
    """
    Get a specific prompt by name.
    
    Args:
        prompt_name (str): Name of the prompt (e.g., 'emotions_correlations_short')
    
    Returns:
        str: The prompt text
    """
    return prompts.get(prompt_name, "Prompt not found")

# Function to list all available prompts
def list_prompts():
    """List all available prompt names."""
    return list(prompts.keys())

# Function to get all versions of a specific prompt type
def get_prompt_versions(base_name):
    """
    Get all versions of a specific prompt type.
    
    Args:
        base_name (str): Base name like 'emotions_correlations_long' or 'relations_no_correlations_long'
    
    Returns:
        dict: Dictionary with version numbers as keys and prompts as values
    """
    versions = {}
    for key in prompts.keys():
        if key.startswith(base_name):
            if key.endswith('_v1'):
                versions['v1'] = prompts[key]
            elif key.endswith('_v2'):
                versions['v2'] = prompts[key]
            elif key.endswith('_v3'):
                versions['v3'] = prompts[key]
            elif key == base_name + '_v1':  # Handle exact match for v1
                versions['v1'] = prompts[key]
    return versions