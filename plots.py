from parsedata import *
import matplotlib.dates as md
import matplotlib.patches as mpatches

def plotNumber(date, value, data, title):
    '''
    fig, ax = plt.subplots(figsize=(15, 7))

    # specify the position of the major ticks at the beginning of the week
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))
    # specify the format of the labels as 'year-month-day'
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    # (optional) rotate by 90° the labels in order to improve their spacing
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    sns.lineplot(ax=ax, x=date, y=value, hue='Day', data=data).set_title(title)
    #sns.barplot(ax=ax, y='Precipitation', data=data)
    plt.show()
    '''

    #fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:green'
    #ax1.set_title('Precipitation and Cyclists at Brooklyn Bridge', fontsize=16)
    #ax1.set_xlabel('Date', fontsize=16)
   #ax1.set_ylabel('Number of Cyclists', fontsize=16, color=color)
    sns.barplot(x='Precipitation', y='Total',data=data, palette='summer')

   # ax1.tick_params(axis='y')
   # ax2 = ax1.twinx()

    color = 'tab:red'
    #ax2.set_ylabel('Percipitation (mm2)', fontsize=16, color=color)
    sns.relplot(x=date.dt.month_name(), y=value, data=data, color=color)
   # ax2.tick_params(axis='y', color=color)
    plt.show()
    return

def plotMulti(date, data, title):
    fig, ax = plt.subplots(figsize=(15, 7))
    # specify the position of the major ticks at the beginning of the week
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))
    # specify the format of the labels as 'year-month-day'
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    # (optional) rotate by 90° the labels in order to improve their spacing
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    sns.lineplot(ax=ax, x=date, y='Brooklyn Bridge', color='red',data=data).set_title(title)
    sns.lineplot(ax=ax, x=date, y='Manhattan Bridge',color='green', data=data)
    sns.lineplot(ax=ax, x=date, y='Williamsburg Bridge',color='blue', data=data)
    sns.lineplot(ax=ax, x=date, y='Queensboro Bridge', color='black',data=data)
    ax.set(xlabel='Date', ylabel='Number of cyclists')
    plt.legend(labels=['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge','Queensboro Bridge'])
    plt.legend()
    plt.show()

def plotBridge(data):

    df = data
    #sns.barplot(x='Precipitation', y='Total', data=data, palette='summer')
    fig, ax = plt.subplots(figsize=(18, 7))
    Date = df['Date']
    bridges = df[['Brooklyn Bridge', 'Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge']]
    c = sns.pointplot(data=df, x=Date.dt.month_name(), y='Brooklyn Bridge', color="b",
                      label="Brooklyn Bridge")
    d = sns.pointplot(data=df, x=Date.dt.month_name(), y='Manhattan Bridge', color="r",
                      label="Manhattan Bridge")
    r = sns.pointplot(data=df, x=Date.dt.month_name(), y='Williamsburg Bridge', color="g",
                      label="Williamsburg Bridge")
    w = sns.pointplot(data=df, x=Date.dt.month_name(), y='Queensboro Bridge', color="black",
                      label="Queensboro Bridge")
    b_patch = mpatches.Patch(color='b', label='Brooklyn Bridge')
    r_patch = mpatches.Patch(color='r', label='Manhattan Bridge')
    g_patch = mpatches.Patch(color='g', label='Williamsburg Bridge')
    black_patch = mpatches.Patch(color='#000000', label='Queensboro Bridge')
    ax.set_title('Number of cyclists by Bridge', fontsize=22, y=1.015)
    ax.set_xlabel('month-day-year', labelpad=16)
    ax.set_ylabel('# of people', labelpad=16)
    ax.set(yscale="log")
    t = plt.xticks(rotation=45)
    ax.legend(handles=[ g_patch,r_patch, black_patch, b_patch])
    plt.show()

