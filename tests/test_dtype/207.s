
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_2n11>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_lshl_b32 s4, s15, 1                                      // 000000001708: 8404810F
	s_mul_hi_i32 s6, s14, 0x55555556                           // 00000000170C: 9706FF0E 55555556
	s_ashr_i32 s5, s4, 31                                      // 000000001714: 86059F04
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001718: BF870009
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
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001748: BF8704C9
	s_cmp_lt_i32 s6, 1                                         // 00000000174C: BF048106
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_cselect_b32 s2, s2, 0                                    // 000000001754: 98028002
	s_add_i32 s6, s14, 2                                       // 000000001758: 8106820E
	s_mul_hi_i32 s7, s6, 0x55555556                            // 00000000175C: 9707FF06 55555556
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001764: BF870499
	s_lshr_b32 s8, s7, 31                                      // 000000001768: 85089F07
	s_add_i32 s7, s7, s8                                       // 00000000176C: 81070807
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001770: BF870499
	s_mul_i32 s7, s7, 3                                        // 000000001774: 96078307
	s_sub_i32 s6, s6, s7                                       // 000000001778: 81860706
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000177C: BF8704A9
	s_cmp_lt_i32 s6, 1                                         // 000000001780: BF048106
	s_cselect_b32 s3, s3, 0                                    // 000000001784: 98038003
	s_add_i32 s2, s3, s2                                       // 000000001788: 81020203
	s_add_u32 s3, s0, s4                                       // 00000000178C: 80030400
	s_addc_u32 s4, s1, s5                                      // 000000001790: 82040501
	s_ashr_i32 s15, s14, 31                                    // 000000001794: 860F9F0E
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s2              // 000000001798: CA100080 00000002
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017A0: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A4: BF870009
	s_add_u32 s0, s3, s0                                       // 0000000017A8: 80000003
	s_addc_u32 s1, s4, s1                                      // 0000000017AC: 82010104
	global_store_b32 v0, v1, s[0:1]                            // 0000000017B0: DC6A0000 00000100
	s_nop 0                                                    // 0000000017B8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017BC: BFB60003
	s_endpgm                                                   // 0000000017C0: BFB00000
