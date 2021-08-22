from telethon import TelegramClient
import configparser
import json
import datetime
import argparse


config = configparser.ConfigParser()
config.read("config.ini")
api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
api_hash = str(api_hash)
phone = config['Telegram']['phone']
username = config['Telegram']['username']
client = TelegramClient(username, api_id, api_hash)

# link = 'https://t.me/AgahMoshaver'
# link = 'https://t.me/betasahm1'
# link = 'https://t.me/codal_ir'
link = 'https://t.me/tsetmc_plus'

async def main():
  # if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get posts from telegram channel')
    parser.add_argument('--channel', type=str, default='betasahm1', help='telegram channel name (like betasahm1)')
    parser.add_argument('--period', type=int, default=700, help='time period for extract post in days')
    args = parser.parse_args()

    link = 'https://t.me/' + args.channel
    output = 'telegram_data/' + args.channel + '.json'
    d = args.period

    print('channel: ' , link , ' period: ' , d , 'days  output_location: ' , output)

    channel_entity = await client.get_entity(link)
    all_messages = {}

    today = datetime.datetime(2021, 7, 1)
    limit_time = today + datetime.timedelta(days=-1*d)

    #offset is used to get data started from specific id
    async for message in client.iter_messages(channel_entity):
        if message.date.replace(tzinfo=None) > today.replace(tzinfo=None):
            continue
        if not message.photo:
          all_messages["news"+str(message.id)] = {"message":message.message, "data":str(message.date)}
          if message.date.replace(tzinfo=None) < limit_time.replace(tzinfo=None):
              break
    print('total messages: ' , len(all_messages))


    #to save data
    with open(output, 'w',encoding='utf-8') as fp:
      json.dump(all_messages, fp, ensure_ascii=False)
    print('messages and dates are stored in ' , output)



with client:
    client.loop.run_until_complete(main())