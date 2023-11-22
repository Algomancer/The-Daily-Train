import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { MetricsService } from './metrics.service';

@Controller('metrics')
export class MetricsController {
  constructor(private metricsService: MetricsService) {}

  @Post()
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: Express.Multer.File) {
    const data = await this.metricsService.processCSV(file);
    return { message: 'File processed successfully', data };
  }
}
