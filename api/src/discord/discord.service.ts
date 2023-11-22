import { Injectable, OnModuleInit } from '@nestjs/common';
import {
  AttachmentBuilder,
  Client,
  GatewayIntentBits,
  TextChannel,
} from 'discord.js';

@Injectable()
export class DiscordService implements OnModuleInit {
  private client: Client;
  private readonly channelId = 'Channel ID HERE';

  constructor() {
    this.client = new Client({
      intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages],
    });
  }

  async onModuleInit(): Promise<void> {
    this.client.on('ready', () => {
      console.log(`Logged in as ${this.client.user.tag}!`);
    });
    await this.client.login('Discord bot token here');
  }

  async postToDiscord(file: Buffer, message: string): Promise<void> {
    try {
      const attachment = new AttachmentBuilder(file, {
        name: 'metrics-report.csv',
      });
      const summaryText = message;

      const channel = await this.client.channels.fetch(this.channelId);
      if (channel instanceof TextChannel) {
        await channel.send({ content: summaryText, files: [attachment] });
      }
    } catch (error) {
      console.error('Error posting to Discord:', error);
    }
  }
}
