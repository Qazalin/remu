
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2368_14_3_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 0x54                                    // 000000001708: 9604FF0F 00000054
	v_mov_b32_e32 v1, 0                                        // 000000001710: 7E020280
	s_ashr_i32 s5, s4, 31                                      // 000000001714: 86059F04
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001718: BF8704D9
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000171C: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s4, s2, s4                                       // 000000001724: 80040402
	s_addc_u32 s5, s3, s5                                      // 000000001728: 82050503
	s_lshl_b32 s2, s14, 1                                      // 00000000172C: 8402810E
	s_ashr_i32 s3, s2, 31                                      // 000000001730: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001738: 84828202
	s_add_u32 s2, s4, s2                                       // 00000000173C: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001740: 82030305
	s_clause 0x2                                               // 000000001744: BF850002
	s_load_b64 s[4:5], s[2:3], null                            // 000000001748: F4040101 F8000000
	s_load_b64 s[6:7], s[2:3], 0x70                            // 000000001750: F4040181 F8000070
	s_load_b64 s[2:3], s[2:3], 0xe0                            // 000000001758: F4040081 F80000E0
	s_waitcnt lgkmcnt(0)                                       // 000000001760: BF89FC07
	v_add_f32_e64 v0, s4, 0                                    // 000000001764: D5030000 00010004
	s_mul_i32 s4, s15, 14                                      // 00000000176C: 96048E0F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001770: BF8704A1
	v_add_f32_e32 v0, s5, v0                                   // 000000001774: 06000005
	s_ashr_i32 s5, s4, 31                                      // 000000001778: 86059F04
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000177C: 84848204
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001780: BF870091
	v_add_f32_e32 v0, s6, v0                                   // 000000001784: 06000006
	v_add_f32_e32 v0, s7, v0                                   // 000000001788: 06000007
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000178C: BF8700A1
	v_add_f32_e32 v0, s2, v0                                   // 000000001790: 06000002
	s_add_u32 s2, s0, s4                                       // 000000001794: 80020400
	v_add_f32_e32 v0, s3, v0                                   // 000000001798: 06000003
	s_addc_u32 s3, s1, s5                                      // 00000000179C: 82030501
	s_ashr_i32 s15, s14, 31                                    // 0000000017A0: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017A4: BF870099
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017A8: 8480820E
	v_mul_f32_e32 v0, 0x3e2aaaab, v0                           // 0000000017AC: 100000FF 3E2AAAAB
	s_add_u32 s0, s2, s0                                       // 0000000017B4: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017B8: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000017BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000017C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017C8: BFB60003
	s_endpgm                                                   // 0000000017CC: BFB00000
