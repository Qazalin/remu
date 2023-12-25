
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_2n1>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_lshl_b32 s4, s15, 1                                      // 000000001708: 8404810F
	s_mul_hi_i32 s6, s14, 0x55555556                           // 00000000170C: 9706FF0E 55555556
	s_ashr_i32 s5, s4, 31                                      // 000000001714: 86059F04
	v_mov_b32_e32 v2, 0                                        // 000000001718: 7E040280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000171C: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s2, s2, s4                                       // 000000001724: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001728: 82030503
	s_lshr_b32 s7, s6, 31                                      // 00000000172C: 85079F06
	s_load_b64 s[2:3], s[2:3], null                            // 000000001730: F4040081 F8000000
	s_add_i32 s6, s6, s7                                       // 000000001738: 81060706
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_mul_i32 s6, s6, 3                                        // 000000001740: 96068306
	s_sub_i32 s6, s14, s6                                      // 000000001744: 8186060E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001748: BF8704D9
	s_cmp_lt_i32 s6, 1                                         // 00000000174C: BF048106
	s_cselect_b32 s6, -1, 0                                    // 000000001750: 980680C1
	s_add_i32 s7, s14, 2                                       // 000000001754: 8107820E
	v_cndmask_b32_e64 v0, 0, 1.0, s6                           // 000000001758: D5010000 0019E480
	s_mul_hi_i32 s8, s7, 0x55555556                            // 000000001760: 9708FF07 55555556
	s_lshr_b32 s9, s8, 31                                      // 000000001768: 85099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000176C: BF870499
	s_add_i32 s8, s8, s9                                       // 000000001770: 81080908
	s_mul_i32 s8, s8, 3                                        // 000000001774: 96088308
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001778: BF8704D9
	s_sub_i32 s6, s7, s8                                       // 00000000177C: 81860807
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	v_fma_f32 v0, v0, s2, 0                                    // 000000001784: D6130000 02000500
	s_cmp_lt_i32 s6, 1                                         // 00000000178C: BF048106
	s_cselect_b32 s2, -1, 0                                    // 000000001790: 980280C1
	v_cndmask_b32_e64 v1, 0, 1.0, s2                           // 000000001794: D5010001 0009E480
	s_add_u32 s2, s0, s4                                       // 00000000179C: 80020400
	s_addc_u32 s4, s1, s5                                      // 0000000017A0: 82040501
	s_ashr_i32 s15, s14, 31                                    // 0000000017A4: 860F9F0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017A8: BF8704A1
	v_fmac_f32_e32 v0, s3, v1                                  // 0000000017AC: 56000203
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017B0: 8480820E
	s_add_u32 s0, s2, s0                                       // 0000000017B4: 80000002
	s_addc_u32 s1, s4, s1                                      // 0000000017B8: 82010104
	global_store_b32 v2, v0, s[0:1]                            // 0000000017BC: DC6A0000 00000002
	s_nop 0                                                    // 0000000017C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017C8: BFB60003
	s_endpgm                                                   // 0000000017CC: BFB00000
