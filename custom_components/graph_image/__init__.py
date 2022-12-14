from homeassistant.config_entries import SOURCE_IMPORT
from homeassistant.core import HomeAssistant
from homeassistant.core import ServiceCall
from homeassistant.components.recorder import history
import homeassistant.util.dt as dt_util

from datetime import timedelta
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import os
# import time

DOMAIN = 'graph_image'

async def async_setup(hass, hass_config):
    if DOMAIN in hass_config and not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_IMPORT}
        ))
    return True


async def async_setup_entry(hass: HomeAssistant, entry):

    def create_graph_image(service: ServiceCall) -> None:

        time_start = dt_util.utcnow() - timedelta(hours=service.data.get('in_time_start', 12)) # дата начала
        time_end = dt_util.utcnow() - timedelta(hours=service.data.get('in_time_end', 0)) # дата окончания
        in_style = service.data.get('in_style', 'Solarize_Light2') # стиль графика
        in_size_ticks = service.data.get('in_size_ticks', 10) # размер подписи осей
        in_rate_ticks = service.data.get('in_rate_ticks', False) # размер подписи осей
        in_linewidth = service.data.get('in_linewidth', 1) # толщина линии
        in_folderfile = service.data.get('in_folderfile', '/config/www/graph.png') # путь и имя для сохранения
        in_linesmooth = service.data.get('in_linesmooth', 1) # уровень сглаживания
        in_lineinterp = service.data.get('in_lineinterp', 'linear_interp') # вид графика
        in_name_notify = service.data.get('in_name_notify', False) # сервис notify для отправки после создания
        in_caption_notify = service.data.get('in_caption_notify', False) # подпись к картинке отправленной через сервис notify
        in_data_notify = service.data.get('in_data_notify', {}) # дополнительные данные для сервиса notify
        entity_ids = service.data["entity_id"]
        plt.style.use(in_style) # стиль графика
        new_caption_notify = ''

        xFmt = mdates.DateFormatter('%m-%d %H:%M', tz=hass.config.time_zone)
        fig = plt.figure()
        fig.set_size_inches(13.5, 7, forward=True) # размеры графика
        ax = fig.add_subplot(1,1,1)
        ax.margins(x=0) # убираем все поля лишние по бокам
        plt.xticks(size = in_size_ticks) # size размер подписи по оси x
        plt.yticks(size = in_size_ticks) # size размер подписи по оси y

        # проходимся по объектам, запрашиваем историю из компонента ha history и формируем график
        for row in entity_ids:
            hist_entity = history.state_changes_during_period(hass, time_start, time_end, row, include_start_time_state=True, no_attributes=True) # запрос истории
            label_name = f"{hist_entity[row][0].attributes['friendly_name']} - {hist_entity[row][0].state}"
            new_caption_notify += f"{label_name}\n"
            x_axis = []
            y_axis = []
            for rec in hist_entity[row]:
                his_state = rec.state
                his_last_updated = rec.last_updated
                try:
                    float(his_state) # для проверки что на входе число, иначе в исключение и продолжаем дальше
                    x_axis.append(his_last_updated)
                    y_axis.append(his_state)
                except Exception:
                    pass
            x_axis = np.array(x_axis)
            y_axis = np.array(y_axis).astype('float32')
            w=np.hanning(in_linesmooth) # сглаживание графика
            y_axis2=np.convolve(w/w.sum(),y_axis,mode='same')

            if in_lineinterp == 'linear_interp':
                ax.plot(x_axis, y_axis2, label=label_name, linewidth = in_linewidth)
            else:
                new_where = 'pre' if in_lineinterp == 'vert_first' else 'post'
                ax.step(x_axis, y_axis2, label=label_name, where = new_where, linewidth = in_linewidth)

        # в зависимости от кол-ва дней, разницы time_end и in_start, частота делений по x или пропускаем и автоматически
        if not in_rate_ticks:
            diff_day = time_end - time_start
            new_interval = 1 if diff_day.days <= 1 else diff_day.days * 2
            ax.xaxis.set_major_locator(mdates.HourLocator(interval = new_interval))

        ax.xaxis.set_major_formatter(xFmt)
        ax.legend(loc=2, ncol=2, shadow = True, fancybox = True, framealpha = 0.5, fontsize = 14) # подпись сенсоров
        fig.autofmt_xdate()
        fig.savefig(in_folderfile, bbox_inches='tight') # сохраняем картинку
        plt.clf()
        ax.cla() 
        fig.clf()

        # отправка картинки через сервис notify
        if in_name_notify:
            in_name_notify = in_name_notify.replace('notify.', '')
            caption_notify = in_caption_notify if in_caption_notify else new_caption_notify
            photo = {}
            photo['file'] = in_folderfile
            photo['caption'] = caption_notify[0:999]
            data = {**in_data_notify}
            data['photo'] = photo
            data['parse_mode'] = 'html'
            hass.services.call('notify', in_name_notify, {
                                    "data": data,
                                    "message": caption_notify[0:999]})
            # time.sleep(1)
            # os.remove(in_folderfile)

    hass.services.async_register(DOMAIN, 'create_graph_image', create_graph_image)

    return True