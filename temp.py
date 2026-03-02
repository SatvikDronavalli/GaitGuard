# pip install pyautogui
import time
import random
import pyautogui
import string

# --- Put your text EXACTLY between the triple quotes below ---
TEXT = r"""Ait-Sahalia, Yacine, and Lars Peter Hansen, editors. Handbook of Financial Econometrics: Applications. Elsevier, 2010.

This scholarly handbook provides a comprehensive overview of the mathematical and statistical techniques used in financial markets, including time-series analysis, volatility modeling, and risk management. The central argument is that financial markets can be better understood and predicted through rigorous quantitative methods, particularly econometrics and stochastic modeling. The book demonstrates how these tools are applied in real-world trading, including pricing derivatives and managing portfolios under uncertainty.

This source contributes to my research by explaining the theoretical foundation behind quantitative trading. It helps me understand the advanced mathematics, such as probability theory and statistical inference, that are required in the field. It also highlights how quantitative analysts use data to develop trading strategies, which answers my question about what technical skills are needed and how they are applied in practice.

Chan, Ernest P. Quantitative Trading: How to Build Your Own Algorithmic Trading Business. Wiley, 2009.

In this book, Ernest Chan argues that individual traders can successfully compete in financial markets by using algorithmic and data-driven strategies. He explains key concepts such as mean reversion, momentum strategies, backtesting, and risk control. Chan emphasizes that successful quantitative trading relies on disciplined statistical analysis rather than intuition or emotion. The book also outlines the process of building and evaluating trading systems using programming languages like Python.

This source is valuable because it provides a practical perspective on how quantitative traders actually operate. It helps answer my question about what day-to-day work looks like in this career, including coding, testing strategies, and managing risk. Additionally, it introduces tools and workflows used in the industry, giving me insight into how I might prepare for this career through programming and data analysis skills.

Mehta, Aakash. Personal Interview. 2 Mar. 2026.

I selected Aakash Mehta as an interview source because he is a quantitative trading intern at the University of Virginia with direct experience in the field I am researching. As a current student transitioning into professional quantitative trading, he offers relevant and up-to-date insights into both academic preparation and industry expectations. His experience allows him to explain the skills, challenges, and realities of entering this career path.

In the interview, Mehta explained that quantitative trading involves analyzing large datasets to identify patterns that can be used to make profitable trades. He emphasized the importance of strong skills in mathematics, statistics, and programming, particularly Python. He also discussed how internships require applicants to demonstrate problem-solving ability through technical interviews, often involving probability and algorithmic questions. This source contributes to my research by providing a real-world perspective on how to enter the field and what skills are most important, helping me better understand the pathway from college to a career in quantitative trading."""
# Seconds to wait before typing starts (so you can click into the target window)
START_DELAY = 5

# Base typing speed (seconds per character)
BASE_CHAR_DELAY = 0.02

# Human-ish variation
JITTER = 0.03  # adds randomness to delay

# Typo behavior controls
TYPO_PROB = 0.015          # chance per character to introduce a typo (≈ 1–2 per 100 chars)
FIX_TYPO_PROB = 0.65       # when a typo happens, chance we backspace + fix it
PAUSE_PROB = 0.02          # occasional pauses like thinking
PAUSE_RANGE = (0.15, 0.8)  # pause length in seconds

def rand_delay():
    # random delay around BASE_CHAR_DELAY
    return max(0.0, BASE_CHAR_DELAY + random.uniform(-JITTER, JITTER))

def random_typo_char(correct_char: str) -> str:
    # If it's a letter, swap to a nearby-ish random letter; else random printable
    if correct_char.isalpha():
        letters = string.ascii_lowercase if correct_char.islower() else string.ascii_uppercase
        c = random.choice(letters)
        return c if c != correct_char else random.choice(letters)
    if correct_char.isdigit():
        d = random.choice(string.digits)
        return d if d != correct_char else random.choice(string.digits)
    # punctuation/space/newline: pick a plausible mistake
    return random.choice([" ", ".", ",", ";", ":", "-", "_"])

def maybe_pause():
    if random.random() < PAUSE_PROB:
        time.sleep(random.uniform(*PAUSE_RANGE))

def type_like_student(text: str):
    for ch in text:
        maybe_pause()

        # Decide if we introduce a typo for this character (avoid typos on newlines)
        if ch != "\n" and random.random() < TYPO_PROB:
            wrong = random_typo_char(ch)

            # Type the wrong character
            pyautogui.write(wrong, interval=rand_delay())

            # Sometimes fix it, sometimes leave it
            if random.random() < FIX_TYPO_PROB:
                # brief "realize mistake" pause
                time.sleep(random.uniform(0.05, 0.25))
                pyautogui.press("backspace")
                time.sleep(random.uniform(0.02, 0.15))
                pyautogui.write(ch, interval=rand_delay())
            else:
                # leave it wrong, and continue with the next correct character
                # (so the final output includes occasional errors for you to fix later)
                pass
        else:
            # Normal correct typing
            pyautogui.write(ch, interval=rand_delay())

def main():
    print(f"Typing will start in {START_DELAY} seconds. Click where you want the text typed...")
    time.sleep(START_DELAY)
    type_like_student(TEXT)

if __name__ == "__main__":
    main()
