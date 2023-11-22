import { Injectable } from '@nestjs/common';
import { DiscordService } from '../discord/discord.service';

@Injectable()
export class MetricsService {
  constructor(private discordService: DiscordService) {}

  async processCSV(file: Express.Multer.File): Promise<any> {
    if (!file?.buffer) {
      throw new Error('File buffer is undefined');
    }

    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString('en-GB');
    const keyChanges = `
**Todays key changes are 🚀:**
- \`0.0.1\`: Added new feature
- \`0.0.2\`: Fixed bug
    `;

    const summaryMessage = `
    ** 🚂 All aboard The Daily Train! (${formattedDate}):**
    ${keyChanges}`;

    // Pass the file buffer directly to the Discord service
    await this.discordService.postToDiscord(file.buffer, summaryMessage);

    // Return a confirmation message or the parsed data, depending on your requirements
    return { message: 'File processed and sent to Discord successfully.' };
  }
}
