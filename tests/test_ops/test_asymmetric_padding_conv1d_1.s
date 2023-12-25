
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_5_2>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_mov_b32 s9, 0                                            // 000000001618: BE890080
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s6, s6, -8                                       // 000000001620: 8006C806
	s_addc_u32 s7, s7, -1                                      // 000000001624: 8207C107
	s_add_i32 s10, s15, -1                                     // 000000001628: 810AC10F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 00000000162C: BF8704C9
	s_cmp_lt_i32 s10, 1                                        // 000000001630: BF04810A
	s_cselect_b32 s3, -1, 0                                    // 000000001634: 980380C1
	s_cmp_gt_i32 s15, 4                                        // 000000001638: BF02840F
	s_cselect_b32 s8, -1, 0                                    // 00000000163C: 980880C1
	s_or_b32 s3, s8, s3                                        // 000000001640: 8C030308
	s_mov_b32 s8, 0                                            // 000000001644: BE880080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001648: 8B6A037E
	s_cbranch_vccnz 7                                          // 00000000164C: BFA40007 <r_5_2+0x6c>
	s_ashr_i32 s3, s2, 31                                      // 000000001650: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001654: BF870499
	s_lshl_b64 s[12:13], s[2:3], 2                             // 000000001658: 848C8202
	s_add_u32 s12, s6, s12                                     // 00000000165C: 800C0C06
	s_addc_u32 s13, s7, s13                                    // 000000001660: 820D0D07
	s_load_b32 s9, s[12:13], null                              // 000000001664: F4000246 F8000000
	s_load_b32 s11, s[0:1], null                               // 00000000166C: F40002C0 F8000000
	s_cmp_gt_u32 s10, 2                                        // 000000001674: BF08820A
	s_cbranch_scc1 7                                           // 000000001678: BFA20007 <r_5_2+0x98>
	s_mov_b32 s3, 0                                            // 00000000167C: BE830080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001680: BF870499
	s_lshl_b64 s[12:13], s[2:3], 2                             // 000000001684: 848C8202
	s_add_u32 s6, s6, s12                                      // 000000001688: 80060C06
	s_addc_u32 s7, s7, s13                                     // 00000000168C: 82070D07
	s_load_b32 s8, s[6:7], 0x4                                 // 000000001690: F4000203 F8000004
	s_load_b32 s0, s[0:1], 0x4                                 // 000000001698: F4000000 F8000004
	s_waitcnt lgkmcnt(0)                                       // 0000000016A0: BF89FC07
	v_fma_f32 v0, s9, s11, 0                                   // 0000000016A4: D6130000 02001609
	s_ashr_i32 s3, s2, 31                                      // 0000000016AC: 86039F02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000016B0: BF8704B1
	v_fmac_f32_e64 v0, s8, s0                                  // 0000000016B4: D52B0000 00000008
	v_mov_b32_e32 v1, 0                                        // 0000000016BC: 7E020280
	s_lshl_b64 s[0:1], s[2:3], 2                               // 0000000016C0: 84808202
	s_add_u32 s0, s4, s0                                       // 0000000016C4: 80000004
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000016C8: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000016CC: 20000080
	s_addc_u32 s1, s5, s1                                      // 0000000016D0: 82010105
	global_store_b32 v1, v0, s[0:1]                            // 0000000016D4: DC6A0000 00000001
	s_nop 0                                                    // 0000000016DC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016E0: BFB60003
	s_endpgm                                                   // 0000000016E4: BFB00000
