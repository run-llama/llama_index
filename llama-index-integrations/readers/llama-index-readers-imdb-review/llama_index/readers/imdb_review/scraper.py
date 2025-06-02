try:
    import concurrent.futures
    import os
    import re

    import imdb
    import pandas as pd
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
except ImportError:
    print("There is an import error")


def clean_text(text: str) -> str:
    """
    Clean raw text string.

    Args:
        text (str): Raw text to clean.

    Returns:
        str: cleaned text.

    """
    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub("Was this review helpful? Sign in to vote.", "", text)
    text = re.sub("Permalink", "", text)
    text = re.sub(r"\.\.\.", "", text)
    text = re.sub(r"\.\.", "", text)
    text = re.sub('""', "", text)
    # Use re.search to find the match in the sentence
    text = re.sub(r"\d+ out of \d+ found this helpful", "", text)
    return text.strip()  # strip white space at the ends


def scrape_data(revs):
    """
    Multiprocessing function to get the data from the IMDB reviews page.

    Args:
        revs (selenium element): element for all the reviews

    Returns:
        date (str): The date of the review
        contents (str): the review of the movie
        rating (str): The ratinng given by the user
        title (str): the title of the review
        link(str): the link of the review

    """
    try:
        spoiler_btn = revs.find_element(By.CLASS_NAME, "ipl-expander")
        spoiler_btn.click()
        spoiler = True
        contents = revs.find_element(By.CLASS_NAME, "content").text
    except NoSuchElementException:
        spoiler = False
        # try:
        #     footer.click()
        # except: pass
        contents = revs.find_element(By.CLASS_NAME, "content").text
        if contents == "":
            contents = revs.find_element(By.CLASS_NAME, "text show-more__control").text

    try:
        title = revs.find_element(By.CLASS_NAME, "title").text.strip()
    except NoSuchElementException:
        title = ""
    try:
        link = revs.find_element(By.CLASS_NAME, "title").get_attribute("href")
    except NoSuchElementException:
        link = ""
    try:
        rating = revs.find_element(
            By.CLASS_NAME, "rating-other-user-rating"
        ).text.split("/")[0]
    except NoSuchElementException:
        rating = 0.0

    re.sub("\n", " ", contents)
    re.sub("\t", " ", contents)
    contents.replace("//", "")
    date = revs.find_element(By.CLASS_NAME, "review-date").text
    contents = clean_text(contents)
    return date, contents, rating, title, link, spoiler


def process_muted_text(mute_text: str) -> (float, float):
    """
    Post processing the muted text.

    Args:
        mute_text (str): text on how many people people found it helpful

    Returns:
        found_helpful (float): Number of people found the review helpful
        total (float): Number of people voted

    """
    found_helpful, total = 0, 0
    pattern = r"(\d+)\s*out\s*of\s*(\d+) found this helpful"
    match = re.search(pattern, mute_text)
    if match:
        # Extract the two numerical figures
        found_helpful = match.group(1)
        total = match.group(2)
    return found_helpful, total


def main_scraper(
    movie_name: str,
    webdriver_engine: str = "edge",
    generate_csv: bool = False,
    multithreading: bool = False,
    max_workers: int = 0,
    reviews_folder: str = "movie_reviews",
):
    """
    The main helper function to scrape data.

    Args:
        movie_name (str): The name of the movie along with the year
        webdriver_engine (str, optional): The webdriver engine to use. Defaults to "edge".
        generate_csv (bool, optional): whether to save the dataframe files. Defaults to False.
        multiprocessing (bool, optional): whether to use multithreading
        max_workers (int, optional): number of workers for multithreading application
    Returns:
        reviews_date (List): list of dates of each review
        reviews_title (List): list of title of each review
        reviews_comment (List): list of comment of each review
        reviews_rating (List):  list of ratings of each review
        reviews_link (List):  list of links of each review

    """
    if multithreading:
        assert max_workers > 0, (
            "If you are using multithreading, then max_workers should be greater than 1"
        )

    ia = imdb.Cinemagoer()
    movies = ia.search_movie(movie_name)
    movie_name = movies[0].data["title"] + " " + str(movies[0].data["year"])

    assert movie_name != "", "Please check the movie name that you passed"
    print(
        f"Scraping movie reviews for {movie_name}. If you think it is not the right one, the best practice is to pass the movie name and year"
    )
    movie_id = movies[0].movieID
    movie_link = f"https://www.imdb.com/title/tt{movie_id}/reviews/?ref_=tt_ql_2"
    if webdriver_engine == "edge":
        driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))
    elif webdriver_engine == "google":
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    elif webdriver_engine == "firefox":
        driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))

    driver.get(movie_link)
    driver.maximize_window()

    driver.execute_script("return document.body.scrollHeight")
    num_reviews = driver.find_element(
        By.XPATH, '//*[@id="main"]/section/div[2]/div[1]/div/span'
    ).text
    print(f"Total number of reviews are: {num_reviews}")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight-250);")
        try:
            load_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "ipl-load-more__button"))
            )
            load_button.click()
        except Exception:
            print("Load more operation complete")
            break

    driver.execute_script("window.scrollTo(0, 100);")

    rev_containers = driver.find_elements(By.CLASS_NAME, "review-container")
    muted_text = driver.find_elements(By.CLASS_NAME, "text-muted")
    muted_text = [process_muted_text(mtext.text) for mtext in muted_text]
    assert len(rev_containers) == len(muted_text), "Same length"

    reviews_date = []
    reviews_comment = []
    reviews_rating = []
    reviews_title = []
    reviews_link = []
    reviews_found_helpful = []
    reviews_total_votes = []
    reviews_if_spoiler = []
    if multithreading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(scrape_data, rev_containers)

        for res in results:
            date, contents, rating, title, link, found_helpful, total, spoiler = res
            reviews_date.append(date)

            reviews_comment.append(contents)
            reviews_rating.append(rating)
            reviews_title.append(title)
            reviews_link.append(link)
            reviews_found_helpful.append(found_helpful)
            reviews_total_votes.append(total)
            reviews_if_spoiler.append(spoiler)
    else:
        for idx, rev in enumerate(rev_containers):
            date, contents, rating, title, link, spoiler = scrape_data(rev)
            found_helpful, total = muted_text[idx]
            reviews_date.append(date)
            # time.sleep(0.2)
            reviews_comment.append(contents)
            reviews_rating.append(float(rating))
            reviews_title.append(title)
            reviews_link.append(link)
            reviews_found_helpful.append(float(found_helpful))
            reviews_total_votes.append(float(total))
            reviews_if_spoiler.append(spoiler)

    print(f"Number of reviews scraped: {len(reviews_date)}")
    if generate_csv:
        os.makedirs(reviews_folder, exist_ok=True)
        df = pd.DataFrame(
            columns=[
                "review_date",
                "review_title",
                "review_comment",
                "review_rating",
                "review_helpful",
                "review_total_votes",
                "reviews_if_spoiler",
            ]
        )

        df["review_date"] = reviews_date
        df["review_title"] = reviews_title
        df["review_comment"] = reviews_comment
        df["review_rating"] = reviews_rating
        df["review_link"] = reviews_link
        df["review_helpful"] = reviews_found_helpful
        df["review_total_votes"] = reviews_total_votes
        df["reviews_if_spoiler"] = reviews_if_spoiler
        df.to_csv(f"{reviews_folder}/{movie_name}.csv", index=False)

    return (
        reviews_date,
        reviews_title,
        reviews_comment,
        reviews_rating,
        reviews_link,
        reviews_found_helpful,
        reviews_total_votes,
        reviews_if_spoiler,
    )
