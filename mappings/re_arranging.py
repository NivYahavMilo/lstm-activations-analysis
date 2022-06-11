def rearrange_clips(df, where: str = 'columns', with_testretest: bool = False):
    clips_order = ['twomen_clips', 'bridgeville_clips', 'pockets_clips', 'overcome_clips', 'testretest_clips',
                   'inception_clips',
                   'socialnet_clips', 'oceans_clips', 'flower_clips', 'hotel_clips', 'garden_clips', 'dreary_clips',
                   'homealone_clips', 'brokovich_clips', 'starwars_clips', 'twomen_rest_between',
                   'bridgeville_rest_between',
                   'pockets_rest_between', 'overcome_rest_between', 'testretest_rest_between', 'inception_rest_between',
                   'socialnet_rest_between', 'oceans_rest_between', 'flower_rest_between', 'hotel_rest_between',
                   'garden_rest_between', 'dreary_rest_between', 'homealone_rest_between', 'brokovich_rest_between',
                   'starwars_rest_between']

    if not with_testretest:
        clips_order[:] = (clip for clip in clips_order if not clip.startswith("test"))

    if where == 'rows':
        df = df.reindex(clips_order)
    else:
        df = df[clips_order]
    return df
