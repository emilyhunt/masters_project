import tweepy
import time
from typing import Optional


class TweetWriter:

    def __init__(self, auth_dir='/home/emily/documents/realemilyastrokeys'):
        """Creates a connection to the """
        # Read in the keys from my file
        auth_file = open(auth_dir, 'r')
        api_key = auth_file.readline()[:-1]
        api_key_secret = auth_file.readline()[:-1]
        access_token = auth_file.readline()[:-1]
        access_token_secret = auth_file.readline()[:-1]

        # Setup Tweepy authorisation
        self.auth = tweepy.OAuthHandler(api_key, api_key_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)

        # We keep a record of the last tweet I made
        self.last_tweet_id = self.get_my_last_tweet_id()

    def write(self, text: str, file: Optional[str]=None, verbose: bool=True, reply_to: Optional[int]=None) -> None:
        """Checks tweet won't be too long and then writes a tweet.
        Args:
            text (str): text to write
            file (str): name of file to include in tweet. Default is None.
            verbose (bool): controls whether or not
            reply_to (int): ID of tweet to reply to. Default is None. -1 will reply to the last tweet from this account.
        Returns:
            None
        """
        # If set to -1, reply_to should reply to our last tweet automatically:
        if reply_to is -1:
            reply_to = self.last_tweet_id

        try:
            # Decide whether or not we need to tweet a file and update the status
            if file is None:
                self.last_tweet_id = self.api.update_status(text, in_reply_to_status_id=reply_to).id
            else:
                self.last_tweet_id = self.api.update_with_media(file, text, in_reply_to_status_id=reply_to).id

            # Provide user updates if desired
            if verbose:
                print('Tweeted: ' + text)

            # Mention that we tweeted out a file if desired
            if file is not None and verbose:
                print('file also tweeted: ' + file)

        except tweepy.TweepError as reason:
            print('An issue was encountered and the following tweet failed:')
            print(text)

            if file is not None:
                print('with file: ' + file)

            print('This was because: ' + str(reason))
            print('Program execution will continue.')

    def get_my_last_tweet_id(self):
        """Returns the ID of the last tweet sent by the currently authorised account."""
        try:
            return self.api.user_timeline(count=1)[0].id
        except tweepy.TweepError as reason:
            print('Failed to get last tweet id because: ' + str(reason))


def time_now():
    """Returns a string of the current time, not including the date."""
    return time.strftime('%X', time.localtime(time.time()))


def calc_local_time(input_time):
    """Convenience function to convert any times to prettier time strings.

    Args:
        input_time (float): the time.time() in seconds you wish to convert to a nice string.
    """
    return time.strftime('%c', time.localtime(input_time))


def short_time_now(date_only: bool=False, time_only: bool=False):
    """Returns the shortest possible current date and time, useful for defining files to write to."""
    the_time = time.localtime(time.time())

    if date_only:
        return time.strftime('%y-%m-%d', the_time)
    elif time_only:
        return time.strftime('%H-%M-%S')
    else:
        return time.strftime('%y-%m-%d--%H-%M-%S', the_time)


def initial_text(thing_this_is_training: str):
    """Helper function that creates initial text to tweet."""
    return ('Beginning training to ' + thing_this_is_training + '. \nStart time is '
            + time_now())


def update_text(thing_thats_done: str, next_thing: Optional[str]=None):
    """Helper function that creates update text to tweet after finishing something."""
    new_string = time_now() + ' - finished task:' + thing_thats_done + '.'

    # Add extra if there's another thing to mention
    if next_thing is not None:
        new_string += 'Now doing: '

    return new_string


def annoy_me(thing_thats_annoying: str):
    """Helper function that tweets at me."""
    return time_now() + ' - Hey, @emilydoesastro! ' + thing_thats_annoying



if __name__ is '__main__':
    # todo tests

    twitter = TweetWriter()

    twitter.write('This tweet is from a manually initiated test of twitter.py at ' + time_now() + '.')
    twitter.write('Did you know I can tweet images?', file='../plots/18-10-23_my_third_nn_res.png', reply_to=-1)
