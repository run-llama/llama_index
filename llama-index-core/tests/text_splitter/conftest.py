import pytest


@pytest.fixture()
def english_text() -> str:
    return """\
A Curious Beginning

In a quaint little village, nestled deep within a lush, green valley, there lived a \
curious young girl named Lily! She had sparkling blue eyes that glimmered like the \
morning dew—yes, like tiny sapphires embedded in her face. And her golden hair flowed \
like a cascade of sunlight, shimmering in the breeze.

Embarking on Enchanted Journeys

Every day, Lily would embark on new adventures; she was like a butterfly dancing on \
the winds of curiosity. Exploring the Enchanting Forests that surrounded her home was \
her favorite pastime. The trees seemed to whisper secrets to her, their leaves \
rustling with ancient tales.
"""


# There's a pretty big difference between GPT2 and cl100k_base for non-English
# The same text goes from 1178 tokens to 665 tokens.
@pytest.fixture()
def chinese_text() -> str:
    return """\
教育的重要性

教育是人类社会发展的基石，也是培养人才、传承文化的重要途径。它不仅能够提升个体的知识水平，\
还能塑造人的品格和价值观。因此，教育在我们的生活中扮演着不可或缺的角色。

首先，教育有助于拓展我们的视野。通过学习，我们能够了解世界各地的文化、历史和科技进展。\
这不仅丰富了我们的知识，还让我们更加开放和包容。教育使我们能够超越狭隘的个人观点，\
理解不同群体的需求和想法，从而促进社会的和谐与发展。

其次，教育培养了未来的领袖和专业人才。在现代社会，各行各业都需要经过专业的教育培训才能胜任。\
教育系统为学生提供了系统的知识体系和技能，使他们能够在职场中脱颖而出。同时，教育也培养了创新能力和\
问题解决能力，为社会的进步和创新奠定了基础。

此外，教育有助于个人的成长和发展。通过学习，人们能够发展自己的才华和潜力，实现人生目标。教育不仅仅是课堂\
上的知识，还包括了品德教育和社会交往的技巧。它教导我们如何与他人合作、沟通，并在逆境中坚持不懈。\
这些都是人生中宝贵的财富，能够引导我们走向成功之路。

总之，教育是我们个人和社会发展的支柱，它不仅丰富了我们的思想，还培养了我们的人才。我们应该珍视教育，\
为其投入更多的资源和关注，以创造一个更加美好的未来。

希望这篇文章对你有帮助！如果你有其他主题的需求，欢迎随时告诉我。\
"""


@pytest.fixture()
def contiguous_text() -> str:
    return "abcde" * 200


# Devanagari marks the end of a sentence with the danda U+0964, and the end of
# a verse or a section with the double danda U+0965. Neither is a full stop, so
# text in Hindi, Marathi, Sanskrit, Nepali and other languages written in
# Devanagari has no Latin or CJK terminator to split on.
@pytest.fixture()
def hindi_text() -> str:
    return """\
शिक्षा का महत्व

शिक्षा मानव समाज के विकास की नींव है। यह केवल व्यक्ति के ज्ञान को नहीं बढ़ाती, \
बल्कि उसके चरित्र और मूल्यों को भी गढ़ती है। इसलिए हमारे जीवन में शिक्षा की भूमिका \
अपरिहार्य है।

सबसे पहले, शिक्षा हमारा दृष्टिकोण विस्तृत करती है। पढ़ाई के माध्यम से हम दुनिया भर \
की संस्कृतियों, इतिहास और विज्ञान को समझ पाते हैं। इससे हमारा ज्ञान समृद्ध होता है \
और हम अधिक सहिष्णु बनते हैं।

दूसरा, शिक्षा भविष्य के नेता और विशेषज्ञ तैयार करती है। आधुनिक समाज में प्रत्येक \
क्षेत्र को प्रशिक्षित लोगों की आवश्यकता होती है। शिक्षा प्रणाली विद्यार्थियों को \
व्यवस्थित ज्ञान और कौशल देती है, जिससे वे अपने कार्यक्षेत्र में आगे बढ़ सकें।

संक्षेप में, शिक्षा व्यक्ति और समाज दोनों के विकास का आधार है। हमें इसे सहेजना \
चाहिए और इसमें अधिक संसाधन लगाने चाहिए।
"""


@pytest.fixture()
def marathi_text() -> str:
    return """\
शिक्षणाचे महत्त्व

शिक्षण हे मानवी समाजाच्या विकासाचा पाया आहे। ते केवळ ज्ञान वाढवत नाही, तर माणसाचे \
चारित्र्य आणि मूल्येही घडवते। त्यामुळे आपल्या जीवनात शिक्षणाची भूमिका अत्यंत \
मोलाची आहे।

पहिली गोष्ट, शिक्षण आपला दृष्टिकोन विस्तृत करते। अभ्यासातून आपल्याला जगभरातील \
संस्कृती, इतिहास आणि विज्ञान समजते। यामुळे आपले ज्ञान समृद्ध होते आणि आपण अधिक \
सहिष्णू होतो।

दुसरी गोष्ट, शिक्षण उद्याचे नेते आणि तज्ज्ञ तयार करते। आजच्या समाजात प्रत्येक \
क्षेत्राला प्रशिक्षित माणसांची गरज असते। शिक्षणपद्धती विद्यार्थ्यांना व्यवस्थित \
ज्ञान आणि कौशल्य देते, ज्यामुळे ते आपल्या कामात उठून दिसतात।

थोडक्यात, शिक्षण हे व्यक्ती आणि समाज या दोघांच्या विकासाचा आधार आहे। आपण त्याचे \
जतन करावे आणि त्यात अधिक संसाधने गुंतवावी।
"""


@pytest.fixture()
def sanskrit_verse_text() -> str:
    return """\
कर्मण्येवाधिकारस्ते मा फलेषु कदाचन॥
योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय॥
सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते॥
विद्या विनयेन शोभते न च केवलेन पाठनेन॥
"""
