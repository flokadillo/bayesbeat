function eval_beats(res_id, dataset, table, write2db)
	addpath('../SilverBeat/utils');
	if strcmpi(dataset, 'ballroom')
		ann_path = '~/diss/data/beats/ballroom/all';
	elseif strfind(dataset, 'boeck') > 0
		ann_path = '~/diss/data/beats/boeck';
	elseif strfind(dataset, 'hainsworth') > 0
		ann_path = '~/diss/data/beats/hainsworth';
	elseif strfind(dataset, 'smc-mirex') > 0
		ann_path = '~/diss/data/beats/smc-mirex';
	elseif strfind(dataset, '1360') > 0
		ann_path = '~/diss/data/beats/BeatTrackLarge/beat_files';
	end
	evaluate_beats_batch('~/diss/src/matlab/beat_tracking/bayes_beat/results/', res_id, ann_path, write2db, 'table_name1', table, 'database_fln', '~/diss/projects/usul/results/ieee_beat_pf.sqlite');
%	evaluate_beats(['~/diss/src/matlab/beat_tracking/bayes_beat/results/', num2str(res_id)], ann_path, write2db, 'table_name', table);
end
