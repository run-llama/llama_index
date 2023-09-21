import pytest


@pytest.fixture
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
@pytest.fixture
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


@pytest.fixture
def contiguous_text() -> str:
    return "abcde" * 200
